"""The live view of the Sputter main screen while Analyze runs: the package's ``@live`` lines, what the page keeps of
them and what it shows, step by step.

Offline, with invented events and a fake of the package's process (no user data).
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.sputter.application import ClientRun, LiveEvent, LiveRun, live_event, parse_live
from tests.sputter_client_fixtures import FakeClientRunner, write_result
from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner, scan_folder

Q = np.r_[-np.linspace(1.0, 0.1, 10), np.linspace(0.1, 1.0, 10)].astype(np.float32)
AF = np.linspace(0.05, 0.9, 12).astype(np.float32)
CELL_Q = np.linspace(0.05, 1.5, 15)
CELL_AF = np.linspace(0.1, 0.8, 9)


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


def _event(kind: str, step: str, arrays=None, **fields) -> LiveEvent:
    return LiveEvent(kind, step, fields, dict(arrays or {}))


def _frames(chunk: int, first: float, rate: float, done: int) -> LiveEvent:
    trace = [[round(first + 0.05 * i, 3), rate + i] for i in range(4)]
    image = np.full((AF.size, Q.size), rate, dtype=np.float32)
    return _event("frames", "intake", {"q": Q, "af": AF, "rate": image}, chunk=chunk, done=done, total=12,
                  t_first=first, t_last=first + 0.15, trace=trace)


def _window(time_s: float, label: str, radius: float, deviation: float, family: str = "zz_net") -> LiveEvent:
    measured = np.outer(np.linspace(1, 2, CELL_AF.size), np.linspace(3, 1, CELL_Q.size))
    return _event("window", "predict", {"q": CELL_Q, "af": CELL_AF, "measured": measured, "model": 0.9 * measured},
                  family=family, t=time_s, t_rel=time_s - 1.0, h_nom=0.2 * time_s, label=label,
                  entry_kind="refined" if label == "refined" else "candidate", fit_deviation=deviation,
                  values={"R_nm": radius, "H_nm": 2 * radius, "D_nm": 4 * radius, "coverage": 0.3, "gap_nm": None})


INTAKE = (_event("centre", "intake", col=99.5, row=199.5),
          _event("detector", "intake", {"image": np.ones((8, 6), np.float16), "masked": np.zeros((8, 6), np.float16)},
                 binning=4, col=3.0, row=5.0, horizon_row=4.0, frames_summed=12, dead=1, hot=2, shadow=30, kept=900,
                 pixels=1000),
          _frames(1, 0.2, 5.0, 8), _frames(0, 0.0, 1.0, 12),
          _event("imported", "intake", frames=12, t_first=0.0, t_last=0.35, onset=0.1))
PREDICT = (_event("models", "predict", families=["zz_net", "zz_other"], mode="quick"),
           _event("windows", "predict", family="zz_net", total=3, refine=0),
           _window(2.0, "candidate 2", 1.5, 4.0), _window(4.0, "candidate 1", 2.0, 3.0))


# -- the lines ---------------------------------------------------------------------------------------------------------


def test_only_live_lines_are_read_as_events():
    record, arrays = parse_live('@live {"kind":"window","step":"predict","t":2.0,"arrays":"C:/zz/1.npz"}')
    assert record == {"kind": "window", "step": "predict", "t": 2.0} and arrays.name == "1.npz"
    assert parse_live('@live {"kind":"drawn","t":3}')[1] is None
    for line in ("[1/4] intake ...", "@live not json", '@live ["a list"]', '@live {"no": "kind"}', "", None):
        assert parse_live(line) is None
    event = live_event({"kind": "drawn", "step": "present", "t": 3.0})
    assert (event.kind, event.step, event.fields, event.number("t")) == ("drawn", "present", {"t": 3.0}, 3.0)


def test_the_run_keeps_the_frames_the_solutions_and_the_network_chosen(tmp_path):
    run = LiveRun()
    assert [run.feed(event) for event in INTAKE] == ["centre", "detector", "frames", "frames", "imported"]
    assert run.centre == (100.0, 200.0)  # GIMaP's pixel coordinates: the package counts pixel centres from 0
    times, rates = run.trace_points()
    assert times == [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] and rates[:2] == [1.0, 2.0]  # in time, not arrival
    assert run.frames_done == 12 and run.frames_total == 12 and run.frames.get("chunk") == 0
    assert run.feed(_event("unknown kind", "intake")) == ""
    for event in PREDICT:
        run.feed(event)
    assert run.network_index() == (1, 2) and run.solved() == 2 and run.totals == {"zz_net": 3}
    run.feed(_window(4.0, "refined", 1.8, 2.0))  # the refined solution joins its window
    result = run.result(tmp_path)
    assert [w.time_s for w in result.windows] == [2.0, 4.0] and result.windows[1].labels() == ["candidate 1", "refined"]
    assert result.windows[1].chosen().label == "refined"  # the lowest fit deviation, as on the main screen
    assert result.series("R_nm")[1] == [1.5, 1.8] and result.windows[0].since_onset_s == 1.0
    patterns, label, time_s, _event_ = run.latest_patterns()
    assert (label, time_s, patterns.labels) == ("refined", 4.0, ("refined",))
    assert patterns.measured.shape == (CELL_AF.size, CELL_Q.size)
    run.feed(_event("windows", "predict", family="zz_other", total=3, refine=0))
    assert run.network_index() == (2, 2) and run.solved() == 0
    # until the next network has a solution, the view keeps the one before (it does not go blank)
    assert run.result(tmp_path).family == "zz_net" and run.latest_patterns()[2] == 4.0
    run.feed(_window(6.0, "candidate 3", 9.0, 7.0, family="zz_other"))
    assert run.latest_patterns()[2] == 6.0 and run.result(tmp_path).family == "zz_other"  # the network at work
    run.feed(_event("chosen", "predict", family="zz_net", deviations={"zz_net": 3.0, "zz_other": 5.5}))
    assert run.chosen == "zz_net" and run.result(tmp_path, run.shown_family()).family == "zz_net"
    assert run.latest_patterns()[1:3] == ("refined", 4.0)  # once chosen: the latest window of the network chosen
    run.feed(_event("drawing", "present", total=2))
    run.feed(_event("drawn", "present", t=2.0))
    assert (run.drawing, run.drawn) == (2, 1)


def _progress(phase: str, done: int, total: int, worker: int, family: str = "zz_net", **more) -> LiveEvent:
    return _event("progress", "predict", phase=phase, done=done, total=total, worker=worker, family=family, **more)


def test_the_phases_of_a_network_add_up_the_workers_and_say_how_far_the_step_has_come():
    from src.gimap.features.sputter.application import ClientProgress

    run = LiveRun()
    run.feed(_event("models", "predict", families=["zz_net", "zz_other"], mode="refine"))
    run.feed(_event("windows", "predict", family="zz_net", total=4, refine=8))
    assert run.phase() == "" and run.step_share("predict") == 0.0
    assert run.feed(_progress("features", 1, 4, 11)) == "progress"
    run.feed(_progress("features", 2, 4, 12))
    run.feed(_progress("features", 2, 4, 11))  # each worker counts its own share: the latest count of each is added
    assert run.phase() == "features" and run.phase_count("features") == (4, 4)
    run.feed(_progress("network", 0, 4, 10))
    assert run.phase() == "network" and run.phase_count("network") == (0, 4)
    share = run.step_share("predict")
    run.feed(_progress("network", 4, 4, 10))
    for time_s in (2.0, 4.0):
        run.feed(_window(time_s, "candidate 1", 1.5, 4.0))
    assert run.phase() == "render" and run.step_share("predict") > share
    run.feed(_progress("refine", 0, 4, 11, evaluations=120))
    run.feed(_progress("refine", 1, 4, 12, evaluations=300))
    run.feed(_window(4.0, "refined", 1.8, 2.0))
    assert run.phase() == "refine" and run.refined() == 1 and run.phase_count("refine", key="evaluations") == (420, 4)
    first = run.step_share("predict")
    assert 0 < first < 0.5  # the first of two networks, not refined through
    run.feed(_progress("refine", 0, 4, 11, evaluations=150, current=30, budget=60))  # half of its window at work
    assert run.step_share("predict") > first
    run.feed(_event("windows", "predict", family="zz_other", total=4, refine=8))
    assert 0.5 <= run.step_share("predict") < 1 and run.phase() == ""
    run.feed(_event("chosen", "predict", family="zz_net", deviations={}))
    assert run.step_share("predict") == 1.0
    assert run.feed(_progress("an unknown phase", 1, 2, 11)) == ""
    # the bar of the job status: the step's share as the events tell it (also after the step's first network ended)
    tracker = ClientProgress()
    for line in ("[1/4] intake ...", "      done in 3 s", "[2/4] predict ..."):
        tracker.feed(line)
    assert tracker.fraction == pytest.approx(0.35)
    tracker.within = 0.5
    assert tracker.fraction == pytest.approx(0.35 + 0.45 * 0.5)
    tracker.feed("      done in 9 s")  # the first network: the analysis runs the next one
    assert tracker.fraction == pytest.approx(0.35 + 0.45 * 0.5)
    tracker.feed("[3/4] report ...")
    assert tracker.within is None and tracker.fraction == pytest.approx(0.8)


def test_the_runner_hands_live_lines_over_with_their_arrays(tmp_path):
    from src.gimap.features.sputter.application import ClientPackage
    from src.gimap.features.sputter.infrastructure.client import ClientPackageRunner
    from src.gimap.features.sputter.infrastructure.process import Outcome

    arrays = tmp_path / "live" / "1.npz"
    arrays.parent.mkdir()
    np.savez_compressed(arrays, q=CELL_Q, rate=np.ones(3))
    lines = ["[1/4] intake ...", "@live " + json.dumps({"kind": "frames", "step": "intake", "done": 3,
                                                        "arrays": str(arrays)}),
             "@live " + json.dumps({"kind": "drawn", "step": "present", "arrays": str(tmp_path / "gone.npz")}),
             "      done in 1 s"]

    def process(command, *, on_line, should_stop, env, cwd):
        for line in lines:
            on_line(line)
        return Outcome(0, list(lines))

    seen = []
    runner = ClientPackageRunner(lambda: tmp_path, "python", run=process)
    run = runner.launch(["python"], ClientPackage(tmp_path, "chosen", "", "0" * 64), tmp_path, tmp_path, seen.append,
                        lambda: False)
    assert [type(item).__name__ for item in seen] == ["str", "LiveEvent", "LiveEvent", "str"]
    assert seen[1].kind == "frames" and np.allclose(seen[1].arrays["q"], CELL_Q) and seen[1].number("done") == 3
    assert seen[2].arrays == {}  # its file is gone: the event comes without its picture
    assert run == ClientRun(0, False, ("[1/4] intake ...", "      done in 1 s"))  # the tail of a failure: no events


# -- the page ----------------------------------------------------------------------------------------------------------


class LiveClientRunner(FakeClientRunner):
    """The package's process: its steps with live events, held after the first two windows until ``go`` is set."""

    def __init__(self):
        super().__init__()
        self.go = threading.Event()
        self.held = threading.Event()

    def run(self, request, out_folder, on_line, should_stop) -> ClientRun:
        self.requests.append(request)
        on_line("[1/4] intake ...")
        for event in INTAKE:
            on_line(event)
        on_line("      done in 3 s")
        on_line("[2/4] predict ...")
        for event in PREDICT:
            on_line(event)
        self.held.set()
        self.go.wait(20)
        on_line("      done in 5 s")
        on_line(_event("chosen", "predict", family="zz_net", deviations={"zz_net": 3.0, "zz_other": 5.5}))
        for line in ("[3/4] report ...", "      done in 1 s", "[4/4] present ..."):
            on_line(line)
        write_result(out_folder)
        on_line("status: ok; report: none")
        return ClientRun(0, False, ("status: ok; report: none",))


def _page(tmp_path, client):
    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.infrastructure import ClientFolderReader
    from src.gimap.features.sputter.presentation.page import SputterPage

    service = create_sputter_service(
        runner=FakeRunner(), installer=FakeInstaller(ready=True), inspector=FakeInspector(),
        results_root=lambda: tmp_path / "results",
        client=create_client_service(runner=client, reader=ClientFolderReader(lambda: tmp_path / "user")))
    page = SputterPage(service, task_runner=TaskRunner())
    page.resize(1256, 869)
    return page


def test_the_main_screen_shows_each_step_and_what_it_found_while_analyze_runs(tmp_path):
    from src.gimap.features.sputter.presentation.live_panel import LIVE_PAGE

    client = LiveClientRunner()
    page = _page(tmp_path, client)
    page.show()
    page.set_scan_folder(scan_folder(tmp_path / "scan"))
    _settle(lambda: page.header_ready())
    page.sdd.setText("1000")
    page.alpha.setText("0.3")
    page.sample_rate.setText("0.25")
    page.analyze_button.click()
    _settle(lambda: client.held.is_set())
    _settle(lambda: page.live_steps.state("predict") == "busy" and page._live.solved() == 2, 5)
    page._live_draw()  # what the timer does at most every REDRAW_MS
    assert page.result_stack.currentIndex() == LIVE_PAGE
    steps = page.live_steps
    assert [steps.state(key) for key in ("intake", "predict", "report", "present")] == ["ok", "busy", "pending",
                                                                                       "pending"]
    assert steps.detail("intake") == "12 frames, 0–0.35 s, deposition from 0.1 s"
    assert steps.detail("predict") == "Network 1 of 2: 2 of 3 time windows"
    assert page.live_title.text() == "Step 2 of 4: Find the solutions"
    assert page.live_patterns.titles == ("Measured image", "Model image: Candidate 1")
    assert page.live_caption.full_text() == ("t = 4 s · Candidate 1 · Radius R 2 nm · Height H 4 nm · Distance D 8 nm "
                                             "· Coverage 0.3 · fit deviation 3")
    assert page.live_evolution.series["R_nm"][1] == [1.5, 2.0]
    assert page.live_clock.text().startswith("Running for 00:0")
    # a click on the first step shows what it found; a click on the step at work follows the analysis again
    page._live_step_chosen("intake")
    page._live_draw()
    assert page.live_picture_host.currentIndex() == 0 and page.live_title.text() == "Step 1 of 4: Read the frames"
    assert page.live_caption.full_text() == ("Beam centre (100.0, 200.0) px · 12 frames summed · 100 pixels left out "
                                             "by the mask (found in these frames: 1 dead, 2 hot, 30 of beamstop and "
                                             "holder)")
    assert page.live_frames.titles == ("Detector, summed", "Frames 0.0–0.1 s")
    assert page.live_trace._last[2] == 0.1  # the start of the deposition is marked
    page._live_step_chosen("present")  # not started: nothing changes
    page._live_draw()
    assert page.live_picture_host.currentIndex() == 0
    page._live_step_chosen("predict")
    page._live_draw()
    assert page.live_picture_host.currentIndex() == 1
    # another language while it runs: the live view stays, in the new language
    from src.gimap.app.presentation.i18n import apply_language

    apply_language("zh", [page])
    page.refresh_language()
    assert page.result_stack.currentIndex() == LIVE_PAGE and page.live_title.text() == "第 2 步（共 4 步）：求解结构"
    apply_language("en", [page])
    page.refresh_language()
    assert page.result_stack.currentIndex() == LIVE_PAGE
    client.go.set()
    _settle(lambda: not page.analyzing())
    assert page.result_stack.currentIndex() == 1 and page.client_result is not None  # the result replaces the view
    assert not page._live_clock.isActive() and page._live is None
    page.dispose()
    page.tasks.wait(10)
    page.close()


def test_a_failed_or_stopped_run_shows_again_what_was_shown_before(tmp_path):
    from src.gimap.features.sputter.presentation.live_panel import LIVE_PAGE

    page = _page(tmp_path, FakeClientRunner())
    page.show()
    page.open_client_result(write_result(tmp_path / "before"))
    page._client_tracker = None
    page._live_begin(tmp_path / "out")
    assert page.result_stack.currentIndex() == LIVE_PAGE
    page._live_event(_event("centre", "intake", col=1.5, row=2.5))
    page._live_draw()
    assert page.live_steps.detail("intake") == "Beam centre (2.0, 3.0) px"
    page._live_end()
    assert page.result_stack.currentIndex() == 1 and page.client_result.folder == tmp_path / "before"
    page.dispose()
    page.close()


def test_until_its_first_window_the_view_says_what_the_network_does_and_the_bar_moves(tmp_path):
    from src.gimap.features.sputter.application import ClientProgress

    page = _page(tmp_path, FakeClientRunner())
    page.show()
    tracker = page._client_tracker = ClientProgress()
    for line in ("[1/4] intake ...", "      done in 3 s", "[2/4] predict ..."):
        tracker.feed(line)
    page._live_begin(tmp_path / "out")
    for event in (_event("models", "predict", families=["zz_net", "zz_other"], mode="refine"),
                  _event("windows", "predict", family="zz_net", total=400, refine=8),
                  _progress("features", 120, 400, 1), _progress("features", 80, 400, 2)):
        page._live_event(event)
    page._live_draw()
    line = "Network 1 of 2: Network inputs ready for 200 of 400 time windows"
    assert page.live_steps.detail("predict") == line
    assert page.live_patterns._last == ("message", line + "\nThe measured and the model image of a time window "
                                                          "appear here once the first one is solved.")
    moved = page.analyze_status.progress_bar.value()
    assert moved > 350  # the bar goes on inside the step (0.35 when the frames were read)
    page._live_event(_progress("network", 0, 400, 3))
    page._live_draw()
    assert page.live_steps.detail("predict") == "Network 1 of 2: The network proposes candidates for 400 time windows…"
    page._live_event(_progress("network", 400, 400, 3))
    page._live_event(_window(2.0, "candidate 1", 1.5, 4.0))
    page._live_draw()
    assert page.live_steps.detail("predict") == "Network 1 of 2: 1 of 400 time windows"
    page._live_event(_progress("render", 1, 400, 2, images=4))  # the model images move on between windows
    page._live_draw()
    assert page.live_steps.detail("predict") == "Network 1 of 2: 1 of 400 time windows solved (4 model images computed)"
    page._live_event(_progress("refine", 0, 400, 1, evaluations=1234))  # a worker refines, others may still solve
    page._live_event(_window(2.0, "refined", 1.6, 3.0))
    page._live_draw()
    line = "Network 1 of 2: 1 of 400 time windows solved, 1 refined (1,238 model images computed)"
    assert page.live_steps.detail("predict") == line
    shown = page.live_steps._steps["predict"].detail
    assert shown.text() == line and shown.wordWrap()  # whole on the screen, wrapped, not cut with "…"
    assert page.analyze_status.progress_bar.value() > moved
    from src.gimap.app.presentation.i18n import apply_language

    apply_language("zh", [page])
    page.refresh_language()
    page._live_draw()
    assert page.live_steps.detail("predict") == "网络 1/2：已求解 1 / 400 个时间窗，已精修 1 个（已计算 1,238 幅模型图）"
    apply_language("en", [page])
    page._live_end()
    page.dispose()
    page.close()


def _static_frame(frame: int, radius: float, deviation: float, stage: str = "network") -> LiveEvent:
    measured = np.outer(np.linspace(1, 2, CELL_AF.size), np.linspace(3, 1, CELL_Q.size))
    return _event("frame", "predict", {"q": CELL_Q, "af": CELL_AF, "measured": measured, "model": 0.8 * measured},
                  label="zz_6nm", index=frame - 1, frame=frame, nominal_nm=6.0, stage=stage, fit_deviation=deviation,
                  values={"R_nm": radius, "H_nm": 5.0, "D_nm": 12.0, "coverage": 0.5})


STATIC_EVENTS = (_event("detector", "intake", {"image": np.ones((8, 6), np.float16), "masked": np.zeros((8, 6), np.float16)},
                        label="zz_6nm", role="sample", binning=4, col=3.0, row=5.0, horizon_row=4.0, frames_summed=9,
                        dead=1, hot=2, shadow=0, kept=950, pixels=1000),
                 _event("sample", "intake", {"q": Q, "af": AF, "rate": np.ones((AF.size, Q.size), np.float32)},
                        label="zz_6nm", role="sample", nominal_nm=6.0, frames=9, usable=8,
                        excluded=[["zz_00001.cbf", "512 saturated pixels: direct or specular beam not blocked"]]),
                 _event("solving", "predict", label="zz_6nm", total=4, refine=0),
                 _static_frame(2, 4.0, 2.8), _static_frame(5, 4.4, 3.2))


def test_a_static_run_keeps_the_sample_its_frames_and_the_series(tmp_path):
    from src.gimap.features.sputter.domain import ClientEntry, ClientResult, ClientWindow

    run = LiveRun(kind="static")
    assert [run.feed(event) for event in STATIC_EVENTS] == ["detector", "sample", "solving", "frame", "frame"]
    assert run.detector is not None and run.sample.number("usable") == 8 and run.frames_to_solve == 4
    assert run.frames_solved() == 2 and run.frame_entry(5).label == "frame 5"
    patterns, label, frame, _event_ = run.frame_patterns()
    assert (label, frame, patterns.labels) == ("frame 5", 5.0, ("frame 5",))
    # the series as it was (another sample, and this one analysed before: it is left out) and this sample's frames
    before = ClientResult(tmp_path, windows=(
        ClientWindow(1.0, nominal_nm=4.0, label="zz_4nm", entries=(ClientEntry("model", values={"R_nm": 3.0}),)),
        ClientWindow(2.0, nominal_nm=6.0, label="zz_6nm", entries=(ClientEntry("model", values={"R_nm": 9.9}),))),
        kind="static", x_label="nominal thickness", x_unit="nm")
    result = run.static_result(tmp_path, before, "zz_6nm", 6.0)
    assert result.static and [w.label for w in result.windows] == ["zz_4nm", "zz_6nm frame 2", "zz_6nm frame 5"]
    assert [result.position(w) for w in result.windows] == [4.0, 6.0, 6.0] and result.series("R_nm")[1] == [3.0, 4.0, 4.4]
    run.feed(_static_frame(2, 4.1, 2.0, stage="refined"))  # the refined solution stands for its frame
    assert run.frame_entry(2).values["R_nm"] == 4.1 and run.static_result(tmp_path, None, "zz_6nm", 6.0).windows[0].label \
        == "zz_6nm frame 2"
    run.feed(_event("detector", "intake", {"image": np.ones((8, 6), np.float16)}, role="bare substrate"))
    assert run.reference_detector is not None


class LiveStaticRunner:
    """The static command: its steps with live events, held after the frames until ``go`` is set."""

    def __init__(self, series_root):
        from tests.sputter_client_fixtures import FakeStaticRunner

        self.inner = FakeStaticRunner(series_root)
        self.go, self.held = threading.Event(), threading.Event()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def run(self, request, on_line, should_stop):
        from tests.sputter_client_fixtures import STATIC_LINES

        on_line(STATIC_LINES[0])
        for event in STATIC_EVENTS[:2]:
            on_line(event)
        for line in STATIC_LINES[1:4]:
            on_line(line)
        for event in STATIC_EVENTS[2:]:
            on_line(event)
        self.held.set()
        self.go.wait(20)
        return self.inner.run(request, lambda line: None if line in STATIC_LINES[:4] else on_line(line), should_stop)


def test_after_deposition_the_main_screen_shows_the_sample_its_frames_and_the_series(tmp_path):
    from src.gimap.features.sputter.presentation.live_panel import LIVE_PAGE
    from src.gimap.integrations.state import InMemoryUserPreferencesRepository
    from tests.test_sputter_main_screen import _page as main_page

    runner = LiveStaticRunner(tmp_path / "series")
    page = main_page(tmp_path, static=runner, preferences=InMemoryUserPreferencesRepository())
    page.show()
    frames = tmp_path / "zz frames"
    frames.mkdir()
    page.set_analysis_kind("static")
    assert page.input_stack.currentIndex() == 1 and page.static_frames.isVisible() and not page.scan_picker.isVisible()
    assert page.static_reference.path_edit.placeholderText() == "optional: another piece, no metal"
    page.static_frames.set_path(str(frames))
    page.static_name.setText("zz 6nm")
    page.static_nominal.setText("6")
    page.analyze_button.click()  # After Deposition: Analyze runs the sample
    _settle(lambda: runner.held.is_set())
    _settle(lambda: page._live is not None and page._live.frames_solved() == 2, 5)
    page._live_draw()
    assert page.result_stack.currentIndex() == LIVE_PAGE and page._live.kind == "static"
    steps = page.live_steps
    assert [steps.title(key) for key in ("intake", "predict", "report")] == [
        "Read the frames of the sample", "Find its solution", "Add it to the series"]
    assert not steps._steps["present"].isVisible()
    assert steps.detail("intake") == "8 of 9 frames usable"
    assert steps.detail("predict") == "2 of 4 frames solved (4 of the 8 usable)"  # Quick solves four of them
    assert page.live_title.text() == "Step 2 of 3: Find its solution"
    assert page.live_patterns.titles == ("Measured image", "Model image: Frame 5")
    assert page.live_caption.full_text().startswith("Frame 5 · network · Radius R 4.4 nm")
    assert page.live_evolution.series["R_nm"][1] == [4.0, 4.4]  # every frame at the nominal thickness
    assert page.live_plot_host.isVisible()
    page._live_step_chosen("intake")
    page._live_draw()
    assert page.live_frames.titles == ("Detector, summed", "Usable frames of the sample, summed")
    assert page.live_caption.full_text().startswith("8 of 9 frames usable · no bare substrate")
    assert "left out: zz_00001.cbf (512 saturated pixels" in page.live_caption.full_text()
    runner.go.set()
    _settle(lambda: not page.static_analyzing())
    assert page.result_stack.currentIndex() == 1 and page.client_result.static and page._live is None
    assert page.analyze_status.state_label.property("gimapJobState") == "succeeded"
    page.dispose()
    page.tasks.wait(10)
    page.close()


def test_the_count_rate_line_is_not_drawn_across_frames_not_read_yet():
    from src.gimap.features.sputter.presentation.live_pictures import side_edges, with_gaps

    times, rates = with_gaps([0.0, 0.05, 0.1, 6.4, 6.45], [1, 2, 3, 4, 5])
    assert np.isnan(times[3]) and list(times[[0, 1, 2, 4, 5]]) == [0.0, 0.05, 0.1, 6.4, 6.45]
    edges, columns = side_edges(Q)  # each side keeps its cells, one empty column between them
    assert edges.size == Q.size + 2 and list(columns[9:11]) == [9, 11]
    assert np.isclose(edges[10], -0.05) and np.isclose(edges[11], 0.05)


def test_the_live_view_speaks_chinese():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation import live_panel, live_pictures, live_static
    from src.gimap.features.sputter.presentation.views import sputter_live_view as view

    held = [*live_panel.TEMPLATES, *live_pictures.TEMPLATES, *live_static.STATIC_TEMPLATES, view.LIVE_TITLE,
            view.LIVE_HINT, *(name for _key, name in view.LIVE_STEP_TITLES)]
    assert [template for template in held if template not in ZH] == []
    for template in held:
        fields = sorted(set(__import__("re").findall(r"{(\w+)}", template)))
        assert sorted(set(__import__("re").findall(r"{(\w+)}", ZH[template]))) == fields, template
