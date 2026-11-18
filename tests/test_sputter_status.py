"""What a running Analyze is doing now, in one line, and the joint fit of the whole scan at work on the main screen.

The status line from each kind of live event (with the time left where the pace is known), the job status and the note
line of the main screen while the preliminary result is shown, the joint fit's current solution dashed over the
evolution until the final result replaces it, and the Chinese texts. Invented events and results only.
"""

from __future__ import annotations

import json
import math
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import ClientProgress, LiveEvent, LiveRun
from tests.sputter_client_fixtures import write_result
from tests.test_sputter_main_screen import _close, _page


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def _event(kind: str, step: str = "predict", **fields) -> LiveEvent:
    return LiveEvent(kind, step, fields, {})


def _progress(phase: str, done: int, total: int = 24, worker: int = 1, family: str = "zz_v4", **more) -> LiveEvent:
    return _event("progress", phase=phase, done=done, total=total, worker=worker, family=family, **more)


def _joint(iteration: int, seconds: float, median: float, total: int = 6, family: str = "zz_v4") -> LiveEvent:
    """The joint fit after one iteration: every window's current solution (R grows by 0.1 nm per iteration)."""
    windows = [{"t": t, "fit_deviation": median, "R_nm": 1.0 + 0.1 * iteration + 0.2 * t, "H_nm": 2.0, "D_nm": 5.0,
                "gap_nm": 1.5, "width": 0.2, "coverage": 0.35, "amount_nm": 0.4, "base_nm": 0.0}
               for t in (8.0, 2.0, 4.0)]  # the package's order need not be the time's
    return _event("joint", family=family, iteration=iteration, total=total, seconds=seconds, median=median,
                  windows=windows)


def _line(run: LiveRun, now: float) -> str:
    from src.gimap.features.sputter.presentation.live_status import status_line

    return status_line(run, now)[0]


# -- the line, event by event ------------------------------------------------------------------------------------------


def test_the_status_line_says_what_the_run_does_now_with_counts_and_the_time_left():
    run = LiveRun()
    assert _line(run, 0.0) == ""  # nothing said yet: the page shows the step
    run.feed(_event("centre", "intake", col=10.0, row=20.0), now=0.0)
    assert _line(run, 0.0) == "Reading the frames: finding the beam centre and the pixel mask"
    run.feed(_event("frames", "intake", done=200, total=1200), now=0.0)
    assert _line(run, 0.0) == "Reading frames: 200 of 1,200"  # one chunk: no pace yet
    run.feed(_event("frames", "intake", done=400, total=1200), now=60.0)
    assert _line(run, 60.0) == "Reading frames: 400 of 1,200 · about 4 min left"  # 800 frames at 200 per minute
    run.note_step()  # a step line came last: it says what happens (the page shows it)
    assert _line(run, 60.0) == ""
    run.feed(_event("imported", "intake", frames=1200, t_first=0.0, t_last=60.0), now=61.0)
    assert _line(run, 61.0) == "1,200 frames read; the networks start"
    run.feed(_event("models", families=["zz_p5", "zz_v4"], mode="refine"), now=62.0)
    assert _line(run, 62.0) == "Starting network zz_p5 (the first of 2 to compare)"
    run.feed(_event("windows", family="zz_v4", total=24, refine=0), now=63.0)
    assert _line(run, 63.0) == "Network zz_v4: starting on 24 time windows"
    run.feed(_progress("features", 8, worker=1), now=63.0)
    run.feed(_progress("features", 4, worker=2), now=63.5)
    assert _line(run, 63.5) == "Network zz_v4: reading the network's input, window 12 of 24"
    run.feed(_progress("network", 0), now=64.0)
    assert _line(run, 64.0) == "Network zz_v4: the network proposes candidates for 24 windows"
    run.feed(_progress("render", 1, images=4), now=70.0)
    run.feed(_progress("render", 7, images=28), now=100.0)  # 6 windows in 30 s: 17 left take 85 s
    assert _line(run, 100.0) == ("Network zz_v4: drawing the candidates with the forward model, window 7 of 24 · about "
                                 "1 min left")
    assert _line(run, 140.0) == ("Network zz_v4: drawing the candidates with the forward model, window 7 of 24 · less "
                                 "than a minute left")  # the time since the latest event counts down
    run.feed(_progress("refine", 3), now=150.0)
    assert _line(run, 150.0) == "Network zz_v4: refining the best candidate of each window, 3 of 24"
    run.feed(_event("chosen", family="zz_v4", deviations={"zz_p5": 21.5, "zz_v4": 12.5}), now=151.0)
    assert _line(run, 151.0) == "Comparing the networks: zz_p5 21.5, zz_v4 12.5 → zz_v4"
    run.feed(_event("drawing", "present", total=20), now=152.0)
    for t in range(4):
        run.feed(_event("drawn", "present", t=float(t)), now=153.0 + t)
    assert _line(run, 156.0) == "Drawing the main screen: window 5 of 20 · less than a minute left"


def test_after_the_preliminary_result_the_line_follows_the_refinement_and_the_joint_fit():
    run = LiveRun()
    run.feed(_event("windows", family="zz_v4", total=24, refine=0), now=0.0)
    run.feed(_progress("render", 24, worker=7), now=10.0)
    run.feed(_event("chosen", family="zz_v4", deviations={"zz_v4": 12.5}), now=11.0)
    assert _line(run, 11.0) == "Network zz_v4: every window solved"
    run.feed(_event("preliminary", family="zz_v4", folder="C:/invented/zz_out"), now=12.0)
    assert run.preliminary and _line(run, 12.0) == "Preliminary result shown; the refinement of every window starts"
    run.feed(_event("refine", family="zz_v4", refine=8, iterations=6), now=13.0)
    assert _line(run, 13.0) == "Network zz_v4: refining every window, then the joint fit of the whole scan"
    run.feed(_event("windows", family="zz_v4", total=24, refine=8), now=14.0)  # the network runs again: from nothing
    run.feed(_progress("features", 3, worker=9), now=15.0)
    assert run.phase_count("render") == (0, 0)
    assert _line(run, 15.0) == "Network zz_v4: reading the network's input, window 3 of 24"
    run.feed(_event("joint_stage", family="zz_v4", stage="pool", windows=20, total=6), now=100.0)
    assert _line(run, 100.0) == "Joint fit: preparing (each window tries its neighbours' solutions, 20 windows)"
    run.feed(_event("joint_stage", family="zz_v4", stage="fit", windows=20, total=6), now=100.0)
    assert _line(run, 130.0) == "Joint fit: working on iteration 1 of 6 · 0:30 so far"  # no estimate before the first
    run.feed(_joint(1, 175.0, 12.0), now=290.0)
    assert run.joint_left(290.0) == pytest.approx(175.0 * 5)
    run.feed(_joint(2, 340.0, 11.3), now=472.0)
    # after two iterations: 170 s per iteration, 4 iterations left
    assert run.joint_left(472.0) == pytest.approx(680.0)
    assert _line(run, 472.0) == ("Joint fit: iteration 2 of 6 · median fit deviation 11.3 · 5:40 so far, about 11 min "
                                 "left")
    from src.gimap.features.sputter.presentation.live_status import status_line

    assert status_line(run, 472.0)[1] == pytest.approx(2 / 6)  # the job status's bar: the iterations
    times, values = run.joint_points()
    assert times == [2.0, 4.0, 8.0] and values["R_nm"] == pytest.approx([1.6, 2.0, 2.8])
    run.feed(_joint(6, 1020.0, 9.1), now=1180.0)
    assert _line(run, 1180.0) == "Joint fit: iteration 6 of 6 · median fit deviation 9.1 · 17:00 so far"
    run.feed(_event("joint", family="zz_v4"), now=1181.0)  # without an iteration: not taken
    assert run.joint.number("iteration") == 6
    left = __import__("src.gimap.features.sputter.presentation.live_status", fromlist=["left_words"]).left_words
    assert (left(30), left(150), left(3 * 3600 + 25 * 60), left(None)) == (
        "less than a minute left", "about 2 min left", "about 3 h 25 min left", "")


def test_an_older_package_without_joint_events_still_says_its_iteration():
    run = LiveRun()
    run.feed(_event("windows", family="zz_v4", total=24, refine=8), now=0.0)
    run.feed(_progress("sequence", 2, total=6), now=1.0)
    assert _line(run, 1.0) == "Joint fit: iteration 2 of at most 6"
    static = LiveRun(kind="static")
    static.feed(_event("solving", total=4), now=0.0)
    assert _line(static, 0.0) == ""  # a sample measured after its deposition keeps its step lines


# -- the page ------------------------------------------------------------------------------------------------------------


def _preliminary(tmp_path):
    out = write_result(tmp_path / "out")
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    (out / "analysis.json").write_text(json.dumps(dict(analysis, status="preliminary")), encoding="utf-8")
    return out


def _running(tmp_path):
    page = _page(tmp_path)
    page.show()
    page._client_tracker = ClientProgress()
    page._live_begin(tmp_path / "out")
    for line in ("[1/4] intake ...", "      done in 3 s", "[2/4] predict ..."):
        page._client_event(line)
    return page


def test_the_live_view_and_the_job_status_say_what_happens_now(tmp_path):
    page = _running(tmp_path)
    page._live_draw()
    assert page.live_status.full_text() == "Step 2 of 4: Finding the solutions of every time window…"  # the step
    for event in (_event("models", families=["zz_v4"], mode="refine"), _event("windows", family="zz_v4", total=24),
                  _progress("features", 12)):
        page._client_event(event)
    page._live_draw()
    line = "Network zz_v4: reading the network's input, window 12 of 24"
    assert page.live_status.full_text() == line and page.analyze_status.message_label.text() == line
    assert page.live_status.isVisible() and not page.run_note.isVisible()  # the main screen shows the live view
    assert page.analyze_status.progress_bar.value() > 350  # the steps' bar: the frames are read
    page._client_busy = True
    page.stop_analysis()  # a stop that was asked for is not written over
    page._client_event(_progress("features", 20))
    assert "Stopping" in page.analyze_status.message_label.text()
    _close(page)


def test_the_joint_fit_at_work_is_drawn_over_the_preliminary_result_until_the_final_one(tmp_path):
    page = _running(tmp_path)
    out = _preliminary(tmp_path)
    page._client_event(_event("chosen", family="zz_v4", deviations={"zz_v4": 12.5}))
    page._client_event(_event("preliminary", family="zz_v4", folder=str(out)))
    assert page.result_stack.currentIndex() == 1 and page._preliminary_shown()
    page._client_event(_event("refine", family="zz_v4", refine=8, iterations=6))
    page._live_draw()
    line = "Network zz_v4: refining every window, then the joint fit of the whole scan"
    assert page.run_note.isVisible() and page.run_note.full_text() == line
    assert page.analyze_status.message_label.text() == line
    assert page.analyze_status.progress_bar.maximum() == 0  # busy: no share of this yet
    assert page.evolution.overlay is None and page.evolution.overlay_series == {}
    page._client_event(_event("joint_stage", family="zz_v4", stage="fit", windows=3, total=6))
    page._client_event(_joint(1, 175.0, 12.0))
    tiles = page.tile_values["R_nm"].text()
    assert page.evolution.overlay[2] == "Joint fit, iteration 1 of 6"
    assert page.evolution.overlay_series["R_nm"] == ([2.0, 4.0, 8.0], pytest.approx([1.5, 1.9, 2.7]))
    assert set(page.evolution.overlay_series) == {"R_nm", "H_nm", "D_nm", "gap_nm", "coverage"}
    legend = [text.get_text() for text in page.evolution._axes[0].get_legend().get_texts()]
    assert legend[-1] == "Joint fit, iteration 1 of 6" and legend.count("Joint fit, iteration 1 of 6") == 1
    page._client_event(_joint(2, 340.0, 11.3))  # the next iteration replaces the overlay
    assert page.evolution.overlay[2] == "Joint fit, iteration 2 of 6"
    assert page.evolution.overlay_series["R_nm"][1] == pytest.approx([1.6, 2.0, 2.8])
    assert page.tile_values["R_nm"].text() == tiles  # the tiles keep the solution shown
    page._live_draw()
    note = page.run_note.full_text()
    assert re.fullmatch(r"Joint fit: iteration 2 of 6 · median fit deviation 11\.3 · 5:4\d so far, about 11 min left",
                        note), note
    assert page.analyze_status.message_label.text() == note
    assert page.analyze_status.progress_bar.value() == round(1000 * 2 / 6)
    from src.gimap.app.presentation.i18n import apply_language

    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        page._live_draw()
        assert page.evolution.overlay[2] == "联合精修 第 2/6 次"
        assert re.fullmatch(r"联合精修：第 2/6 次迭代 · 拟合偏差中位数 11\.3 · 已用 5:4\d，约剩 11 分钟",
                            page.run_note.full_text()), page.run_note.full_text()
    finally:
        apply_language("en", roots=(page,))
        page.refresh_language()
    # the final result (it holds the joint fit): the overlay and the note go
    final = page.service.client.open(write_result(tmp_path / "final"))
    page._client_finished(final)
    assert page.evolution.overlay is None and page.evolution.overlay_series == {}
    assert not page.run_note.isVisible() and page.client_result.folder == final.folder
    assert all(not math.isnan(value) for value in page.evolution.series["R_nm"][1][:2])
    _close(page)


def test_without_a_preliminary_result_the_joint_fit_is_drawn_in_the_live_view(tmp_path):
    from src.gimap.features.sputter.presentation.live_panel import LIVE_PAGE

    page = _running(tmp_path)
    for event in (_event("windows", family="zz_v4", total=3),
                  _event("window", family="zz_v4", t=2.0, label="candidate 1", entry_kind="candidate",
                         fit_deviation=4.0, values={"R_nm": 1.0, "H_nm": 2.0}),
                  _event("joint_stage", family="zz_v4", stage="fit", windows=3, total=6), _joint(1, 60.0, 9.0)):
        page._client_event(event)
    assert page.result_stack.currentIndex() == LIVE_PAGE and page.evolution.overlay is None
    assert page.live_evolution.overlay[2] == "Joint fit, iteration 1 of 6"
    page._live_end()
    assert page.live_evolution.overlay is None
    _close(page)


def test_the_status_texts_speak_chinese():
    from src.gimap.app.presentation.i18n import translate
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation import live_status

    assert [template for template in live_status.TEMPLATES if template not in ZH] == []
    for template in live_status.TEMPLATES:
        fields = sorted(set(re.findall(r"{(\w+)}", template)))
        assert sorted(set(re.findall(r"{(\w+)}", ZH[template]))) == fields, template
    assert translate("Reading frames: {done} of {total}", "zh") == "正在读入帧：{done} / {total}"
