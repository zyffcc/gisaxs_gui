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
    assert run.network_index() == (2, 2) and run.solved() == 0 and run.result(tmp_path) is None
    run.feed(_window(6.0, "candidate 3", 9.0, 7.0, family="zz_other"))
    assert run.latest_patterns()[2] == 6.0  # the network at work
    run.feed(_event("chosen", "predict", family="zz_net", deviations={"zz_net": 3.0, "zz_other": 5.5}))
    assert run.chosen == "zz_net" and run.result(tmp_path, run.shown_family()).family == "zz_net"
    assert run.latest_patterns()[1:3] == ("refined", 4.0)  # once chosen: the latest window of the network chosen
    run.feed(_event("drawing", "present", total=2))
    run.feed(_event("drawn", "present", t=2.0))
    assert (run.drawing, run.drawn) == (2, 1)


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


def test_the_count_rate_line_is_not_drawn_across_frames_not_read_yet():
    from src.gimap.features.sputter.presentation.live_pictures import side_edges, with_gaps

    times, rates = with_gaps([0.0, 0.05, 0.1, 6.4, 6.45], [1, 2, 3, 4, 5])
    assert np.isnan(times[3]) and list(times[[0, 1, 2, 4, 5]]) == [0.0, 0.05, 0.1, 6.4, 6.45]
    edges, columns = side_edges(Q)  # each side keeps its cells, one empty column between them
    assert edges.size == Q.size + 2 and list(columns[9:11]) == [9, 11]
    assert np.isclose(edges[10], -0.05) and np.isclose(edges[11], 0.05)


def test_the_live_view_speaks_chinese():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation import live_panel, live_pictures
    from src.gimap.features.sputter.presentation.views import sputter_live_view as view

    held = [*live_panel.TEMPLATES, *live_pictures.TEMPLATES, view.LIVE_TITLE, view.LIVE_HINT,
            *(name for _key, name in view.LIVE_STEP_TITLES)]
    assert [template for template in held if template not in ZH] == []
    for template in held:
        fields = sorted(set(__import__("re").findall(r"{(\w+)}", template)))
        assert sorted(set(__import__("re").findall(r"{(\w+)}", ZH[template]))) == fields, template
