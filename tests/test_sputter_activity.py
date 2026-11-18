"""Is the analysis package working? Its process tree sampled while it runs, and what the page says from that.

A pool whose worker process ended waits for ever without computing anything: the samples tell it from a package at
work, the live view says so after a minute with the last line of the step's log and offers to stop it. Lines without
a count of their own say how long the run has been at them. Invented processes, lines and events only.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import STALL_S, ActivityWatch, ClientActivity, ClientPackage, LiveEvent
from src.gimap.features.sputter.application import LiveRun
from src.gimap.features.sputter.infrastructure import ClientPackageRunner, Outcome, run_streaming
from src.gimap.features.sputter.infrastructure.client import last_log_line
from src.gimap.features.sputter.infrastructure.process import TreeSample, TreeWatch

psutil = pytest.importorskip("psutil")

LOST_WORKER = """
import multiprocessing as mp


def work(_):
    raise SystemExit("zz_label: the worker gave up")


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(1) as pool:
        pool.map(work, [1])
"""
"""A pool whose only worker ends on SystemExit: the pool starts another one and ``map`` waits for ever."""

BUSY = "import time\nend = time.monotonic() + 30\nwhile time.monotonic() < end:\n    pass\n"


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def _event(kind: str, step: str = "predict", **fields) -> LiveEvent:
    return LiveEvent(kind, step, fields, {})


# -- the process tree --------------------------------------------------------------------------------------------------


def test_a_pool_that_lost_its_worker_is_seen_waiting_and_can_be_stopped(tmp_path):
    script = tmp_path / "lost_worker.py"
    script.write_text(LOST_WORKER, encoding="utf-8")
    samples, started = [], time.monotonic()

    def should_stop() -> bool:  # two quiet samples in a row once the pool runs, or the test gives up
        quiet = len(samples) >= 3 and samples[-1].busy == 0 and samples[-2].busy == 0
        return quiet or time.monotonic() - started > 40

    outcome = run_streaming([sys.executable, str(script)], on_line=lambda _line: None, should_stop=should_stop,
                            on_sample=samples.append, sample_s=0.5)
    assert outcome.stopped and time.monotonic() - started < 40  # stopped because it waited, not by the deadline
    assert samples[-1].processes >= 2  # the script and the worker the pool started instead of the lost one


def test_a_computing_process_counts_as_at_work(tmp_path):
    script = tmp_path / "busy.py"
    script.write_text(BUSY, encoding="utf-8")
    samples = []
    run_streaming([sys.executable, str(script)], on_line=lambda _line: None,
                  should_stop=lambda: len(samples) >= 3, on_sample=samples.append, sample_s=0.4)
    assert any(sample.busy >= 1 for sample in samples[1:])


def test_a_process_that_is_gone_gives_no_sample():
    watch = TreeWatch(2 ** 30)  # no such process
    assert watch.sample() is None and watch.sample() is None


# -- the runner: samples with the last line of the step's log ---------------------------------------------------------


def test_last_log_line_reads_the_end_of_the_step_log_written_last(tmp_path):
    assert last_log_line(tmp_path) == "" and last_log_line(tmp_path / "missing") == ""
    older, newer = tmp_path / "analyze_intake.log", tmp_path / "analyze_predict.log"
    older.write_text("done\n", encoding="utf-8")
    newer.write_text("first\n" + "x" * 9000 + "\nzz_label: the worker gave up\n\n  \n", encoding="utf-8")
    os.utime(older, (time.time() - 60, time.time() - 60))
    assert last_log_line(tmp_path) == "zz_label: the worker gave up"
    newer.write_text("y" * 500 + "\n", encoding="utf-8")
    assert last_log_line(tmp_path) == "y" * 300


def test_the_runner_hands_samples_on_with_the_log_line_while_nothing_works(tmp_path):
    (tmp_path / "analyze_predict.log").write_text("zz_label: the worker gave up\n", encoding="utf-8")

    def process(command, *, on_line, should_stop, env=None, cwd=None, on_sample=None, **_more):
        on_line("[2/4] predict ...")
        on_sample(TreeSample(processes=18, busy=16))
        on_sample(TreeSample(processes=3, busy=0))
        return Outcome(0, [])

    seen = []
    runner = ClientPackageRunner(lambda: tmp_path, "python", run=process)
    runner.launch(["python"], ClientPackage(tmp_path, "chosen", "", "0" * 64), tmp_path, tmp_path, seen.append,
                  lambda: False, logs=tmp_path)
    assert seen == ["[2/4] predict ...", ClientActivity(18, 16, ""),
                    ClientActivity(3, 0, "zz_label: the worker gave up")]


# -- what the samples mean ---------------------------------------------------------------------------------------------


def test_the_package_counts_as_stopped_only_after_a_minute_without_work_or_news():
    watch = ActivityWatch(started=100.0)
    assert watch.busy() is None and watch.quiet(500.0) is None and not watch.stalled(500.0)  # no samples: no claim
    watch.feed(ClientActivity(18, 16), now=105.0)
    assert watch.busy() == 16 and not watch.stalled(105.0 + STALL_S - 1)
    watch.feed(ClientActivity(3, 0, "zz_label: the worker gave up"), now=110.0)
    assert not watch.stalled(105.0 + STALL_S - 1)  # quiet since the last work, not since the last sample
    assert watch.stalled(105.0 + STALL_S) and watch.quiet(200.0) == pytest.approx(95.0)
    assert watch.log_line() == "zz_label: the worker gave up"
    watch.news(190.0)  # a line or an event of the package
    assert not watch.stalled(200.0) and watch.stalled(190.0 + STALL_S)
    watch.feed(ClientActivity(18, 2), now=300.0)  # at work again
    assert not watch.stalled(310.0)


def test_lines_without_a_count_say_how_long_the_run_has_been_at_them():
    from src.gimap.features.sputter.presentation.live_status import status_line

    run = LiveRun()
    run.feed(_event("models", families=["zz_v4", "zz_v5"], mode="refine"), now=100.0)
    first = "Starting network zz_v4 (the first of 2 to compare)"
    assert status_line(run, 105.0)[0] == first  # under 10 s: nothing added
    assert status_line(run, 175.0)[0] == first + " · 1:15 so far"
    run.feed(_event("windows", family="zz_v4", total=24), now=180.0)
    run.feed(_event("progress", phase="network", done=0, total=24, worker=1, family="zz_v4"), now=181.0)
    run.feed(_event("progress", phase="network", done=0, total=24, worker=1, family="zz_v4"), now=190.0)
    assert status_line(run, 223.0)[0] == "Network zz_v4: the network proposes candidates for 24 windows · 0:42 so far"
    run.feed(_event("progress", phase="render", done=3, total=24, worker=1, family="zz_v4"), now=230.0)
    assert "so far" not in status_line(run, 300.0)[0]  # a count: its own pace says how long it still needs


def test_the_line_says_when_the_package_has_stopped_working():
    from src.gimap.features.sputter.presentation.live_status import status_line

    run = LiveRun()
    run.watch.started = 0.0
    run.feed(_event("models", families=["zz_v4"], mode="refine"), now=1.0)
    run.watch.feed(ClientActivity(3, 0), now=10.0)
    assert status_line(run, 70.0)[0] == "No progress for 1:10: the analysis package's processes are all waiting"


# -- the page ----------------------------------------------------------------------------------------------------------


def test_the_live_view_says_how_many_processes_work_and_when_none_does_offers_to_stop(tmp_path):
    from src.gimap.features.sputter.application import ClientProgress
    from tests.test_sputter_main_screen import _close, _page

    page = _page(tmp_path)
    try:
        page.show()
        page._client_tracker = ClientProgress()
        page._client_busy = True
        page._live_begin(tmp_path / "out", workers=16)
        for line in ("[1/4] intake ... reused the imported scan", "[2/4] predict ..."):
            page._client_event(line)
        page._client_event(ClientActivity(18, 16))
        page._live_tick()
        assert page.live_clock.text().endswith("Processes: 16 · 16 at work now")
        assert not page.live_stall_note.isVisible()

        page._client_event(ClientActivity(3, 0, "zz_label: the worker gave up"))
        watch = page._live.watch
        watch.started = watch.worked_at = watch.news_at = time.monotonic() - 95.0  # a minute and a half ago
        page._live_tick()
        assert page.live_clock.text().endswith("Processes: 16 · none at work now")
        note = page.live_stall_note.message_label.text()
        assert page.live_stall_note.isVisible() and note.startswith("Nothing has happened for 1:35")
        assert note.endswith("Last line of its log: zz_label: the worker gave up")
        line = "No progress for 1:35: the analysis package's processes are all waiting"
        assert page.live_status.full_text() == line and page.analyze_status.message_label.text() == line
        assert page.live_stall_note.details_button.text() == "Stop Analysis"

        page.live_stall_note.details_button.click()
        assert page._client_stop  # the run's process tree ends at the next poll

        page._client_stop = False
        page._client_event(ClientActivity(18, 4))  # at work again: the note goes
        page._live_tick()
        assert not page.live_stall_note.isVisible()
        page._live_end()
        assert not page.live_stall_note.isVisible()
    finally:
        page._client_busy = False
        _close(page)


def test_the_new_texts_have_chinese():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation.live_panel import TEMPLATES as PANEL
    from src.gimap.features.sputter.presentation.live_status import STALLED
    from src.gimap.features.sputter.presentation.views.sputter_live_view import STOP_BUTTON

    for key in (*PANEL, STALLED, STOP_BUTTON):
        assert key in ZH, key
