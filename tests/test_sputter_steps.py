"""The steps of an Analyze run as a tree (live view and Details ▸ Steps), what a step found, the joint fit's progress
inside an iteration, and the thickness of the continuous metal layer.

Invented step events, detail files and results only (``tests/sputter_client_fixtures.py``).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import ClientProgress, LiveEvent, LiveRun, StepTree
from tests.sputter_client_fixtures import WANTED, entry, window, write_result
from tests.test_sputter_main_screen import _close, _page, _tiles


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def _event(kind: str, step: str = "predict", **fields) -> LiveEvent:
    return LiveEvent(kind, step, fields, {})


def _step(step_id: str, state: str, summary=None, detail=None, step: str = "analyze") -> LiveEvent:
    return _event("step", step, id=step_id, state=state, title=step_id, summary=summary, detail=detail)


def _progress(iteration: int, done: int, seconds: float, phase: str = "derivatives", windows: int = 20,
              total: int = 6) -> LiveEvent:
    return _event("joint_progress", family="zz_v4", iteration=iteration, total=total, phase=phase, done=done,
                  windows=windows, seconds=seconds)


def _joint(iteration: int, seconds: float, median: float = 9.0, total: int = 6) -> LiveEvent:
    windows = [{"t": t, "fit_deviation": median, "R_nm": 1.0 + 0.1 * iteration, "H_nm": 2.0, "D_nm": 5.0,
                "gap_nm": 1.5, "coverage": 0.4, "base_nm": 0.2 * t} for t in (2.0, 4.0)]
    return _event("joint", family="zz_v4", iteration=iteration, total=total, seconds=seconds, median=median,
                  windows=windows)


def write_detector(folder) -> None:
    """detector.json + detector.npz as the package writes them (invented: 8 x 10 binned pixels, one masked column)."""
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10), 2.0, dtype=np.float16)
    image[0, 0] = np.nan
    masked = np.zeros((8, 10), dtype=np.float16)
    masked[:, 9] = 1.0
    np.savez_compressed(folder / "detector.npz", image=image, masked=masked)
    doc = {"binning": 4, "col": 4.5, "row": 6.0, "horizon_row": 5.0, "frames_summed": 40, "dead": 3, "hot": 2,
           "shadow": 50, "kept": 900, "pixels": 1000, "arrays": "detector.npz"}
    (folder / "detector.json").write_text(json.dumps(doc), encoding="utf-8")


def write_details(folder) -> None:
    """The other detail files of a run (invented numbers)."""
    folder.mkdir(parents=True, exist_ok=True)
    docs = {
        "reading.json": {"frames_read": 300, "frames_total": 1000, "workers": 4, "seconds": 75.0,
                         "read_s_per_frame": 0.2, "read_share": 0.6, "mb_read": 1500.0, "slow": True,
                         "selection": {"frames_read": 300, "frames": 1000, "windows": 12, "onsets": 1, "step": 2.0,
                                       "max_windows": 40, "rounds": 1},
                         "frames_read_ranges": [[0, 99], [400, 449], [900, 999]]},
        "onset.json": {"times": [0.25, 0.75, 1.25, 1.75], "rate": [1.0, 1.0, 1.6, 2.4], "baseline": 1.0,
                       "base_window": [0.3, 1.0], "onset_rule": 1.1, "onset": 1.2, "how": "given",
                       "pre_window": [0.3, 1.15]},
        "windows.json": {"times": [2.0, 4.0, 6.0], "step": 2.0, "max_windows": 40},
        "compare.json": {"deviations": {"zz_p5": 21.5, "zz_v4": 12.5}, "chosen": "zz_v4"},
    }
    for name, doc in docs.items():
        (folder / name).write_text(json.dumps(doc), encoding="utf-8")
    write_joint(folder, 1)


def write_joint(folder, iterations: int) -> None:
    rows = [{"iteration": k, "seconds": 100.0 * k, "median": 12.0 - k, "sum_dev5": 1.0, "penalty": 0.1,
             "windows": [{"t": t, "fit_deviation": 9.0, "R_nm": 1.0 + k, "H_nm": 2.0, "D_nm": 5.0, "gap_nm": 1.0,
                          "width": 0.2, "coverage": 0.5, "amount_nm": 0.3, "base_nm": 0.1 * k}
                         for t in (2.0, 4.0)]} for k in range(1, iterations + 1)]
    (folder / "joint.json").write_text(json.dumps({"family": "zz_v4", "total": 6, "iterations": rows}),
                                       encoding="utf-8")


# -- the tree -----------------------------------------------------------------------------------------------------------


def test_the_tree_follows_the_step_events_and_the_last_report_counts():
    run = LiveRun()
    assert not run.steps.reported and run.steps.running() == ""
    run.feed(_step("import.centre", "running", step="intake"))
    assert run.steps.reported and run.steps.get("import").state == "running"  # a parent whose child runs runs
    assert run.steps.running() == "import.centre" and run.activity == ""  # a step is not "what happens now"
    run.feed(_step("import.centre", "done", "found in the frames: column 12.5, row 30.0", "detector.json", "intake"))
    run.feed(_step("import", "done", "imported before"))
    run.feed(_event("models", families=["zz_p5", "zz_v4"], mode="refine"))
    ids = [node.id for node in run.steps.ordered()]
    assert ids.index("networks.zz_p5") == ids.index("networks") + 1 and ids.index("networks.zz_v4") < ids.index("compare")
    run.feed(_step("windows", "done", "12 windows", "windows.json"))
    run.feed(_step("windows", "done", "13 windows", "windows.json"))  # once per network: the last counts
    assert run.steps.get("windows").summary == "13 windows"
    run.feed(_step("networks.zz_v4", "running"))
    assert run.steps.get("networks").state == "running" and run.steps.running() == "networks.zz_v4"
    assert run.steps.get("networks.zz_v4").family == "zz_v4"
    assert not run.feed(_step("joint", "unknown")) and not run.feed(_event("step", id="", state="done"))
    record = run.steps.to_record()
    again = StepTree.from_record(json.loads(json.dumps(record)))
    assert [(node.id, node.state, node.summary, node.detail) for node in again.ordered()] == \
        [(node.id, node.state, node.summary, node.detail) for node in run.steps.ordered()]
    assert StepTree.from_record({"steps": "no"}) is None
    run.feed(_step("compare", "done", detail="../outside.json"))
    assert run.steps.get("compare").detail == ""  # a path is never followed


def test_a_result_without_the_record_gets_its_tree_from_its_files():
    analysis = {"family": "zz_v4", "families_tried": {"zz_p5": 21.5, "zz_v4": 12.5}, "joint_fit": {"done": False},
                "steps": [{"step": "intake", "returncode": 0}, {"step": "predict", "returncode": 0},
                          {"step": "report", "returncode": 0}, {"step": "present (main)", "returncode": 1}]}
    tree = StepTree.from_files(analysis, {"detector.json", "detector.npz", "reading.json", "joint.json"})
    states = {node.id: (node.state, node.detail) for node in tree.ordered()}
    assert states["import.centre"] == ("done", "detector.json") and states["import.read"] == ("done", "reading.json")
    assert states["import.onset"] == ("done", "") and states["networks.zz_p5"][0] == "done"
    assert states["joint"] == ("failed", "joint.json") and states["finish.report"][0] == "done"
    assert states["finish.present"][0] == "failed"


# -- the joint fit inside an iteration ------------------------------------------------------------------------------------


def test_the_joint_fit_says_its_iteration_batch_and_time_left():
    from src.gimap.features.sputter.presentation.live_status import status_line

    run = LiveRun()
    run.feed(_event("joint_stage", stage="pool", windows=20, total=6), now=0.0)
    run.feed(_progress(0, 3, 10.0, phase="pool"), now=10.0)
    assert status_line(run, 10.0)[0] == "Joint fit: preparing · 3 of 20 windows · 0:10 so far"
    run.feed(_event("joint_stage", stage="fit", windows=20, total=6), now=50.0)
    run.feed(_progress(0, 4, 2.0, phase="step"), now=52.0)  # the starting point: no time left said yet
    assert status_line(run, 52.0)[0] == "Joint fit: the starting point · 4 of 20 windows · 0:02 so far"
    run.feed(_progress(1, 2, 60.0), now=60.0)
    assert status_line(run, 60.0)[0] == "Joint fit: iteration 1 of 6 · derivatives: 2 of 20 windows · 1:00 so far"
    run.feed(_progress(1, 6, 180.0), now=180.0)
    # 30 s per window: 14 windows left in this iteration and 5 more iterations of 20 windows each
    assert run.joint_batch_left(180.0) == pytest.approx(30 * 14 + 600 * 5)
    words, share = status_line(run, 180.0)
    assert words == "Joint fit: iteration 1 of 6 · derivatives: 6 of 20 windows · 3:00 so far, about 57 min left"
    assert share == pytest.approx(0.9 * 6 / 20 / 6)
    run.feed(_joint(1, 700.0), now=700.0)
    assert status_line(run, 700.0)[0].startswith("Joint fit: iteration 1 of 6 · median fit deviation 9")
    run.feed(_progress(2, 4, 820.0), now=820.0)  # a new batch: its pace from the end of the iteration before
    assert run.joint_seconds_per_window() == pytest.approx(30.0)
    assert run.joint_batch_left(820.0) == pytest.approx(30 * 16 + 700 * 4)
    assert status_line(run, 820.0)[0] == ("Joint fit: iteration 2 of 6 · derivatives: 4 of 20 windows · 13:40 so far, "
                                          "about 55 min left")
    run.feed(_progress(2, 10, 1000.0, phase="step"), now=1000.0)
    assert "trial step: 10 of 20 windows" in status_line(run, 1000.0)[0]
    from src.gimap.app.presentation.i18n import translate

    assert translate("Joint fit: iteration {iteration} of {total} · {phase}: {done} of {windows} windows", "zh") == \
        "联合精修：第 {iteration}/{total} 次迭代 · {phase}：{done}/{windows} 个时间窗"


# -- the live view ----------------------------------------------------------------------------------------------------


def _running(tmp_path):
    page = _page(tmp_path)
    page.show()
    page._client_tracker = ClientProgress()
    out = tmp_path / "out"
    page._live_begin(out, trace_folders=(out / "trace", tmp_path / "data" / "gui_zz.trace"))
    for line in ("[1/4] intake ...",):
        page._client_event(line)
    return page


def test_an_older_package_keeps_the_step_lines(tmp_path):
    page = _running(tmp_path)
    page._client_event(_event("centre", "intake", col=10.0, row=20.0))
    page._live_draw()
    assert not page.live_tree_area.isVisibleTo(page) and page.live_steps.isVisibleTo(page)
    assert page.live_hint.text() == "Click a step that has started to see what it found."
    _close(page)


def test_the_live_view_shows_the_tree_and_what_a_step_found(tmp_path):
    page = _running(tmp_path)
    write_detector(tmp_path / "data" / "gui_zz.trace")  # during the import: beside its cache
    page._client_event(_step("import.centre", "done", "found in the frames: column 12.5, row 30.0", "detector.json",
                             "intake"))
    page._client_event(_step("import.read", "running", step="intake"))
    page._client_event(_event("frames", "intake", done=200, total=1000))
    page._live_draw()
    assert page.live_tree_area.isVisibleTo(page) and not page.live_steps.isVisibleTo(page)
    tree = page.live_tree
    centre, read = tree.row("import.centre"), tree.row("import.read")
    assert centre.mark.text() == "✓" and centre.title.full_text() == "Find the beam centre  ›"
    assert centre.summary.full_text() == "found in the frames: column 12.5, row 30.0"
    assert read.mark.text() == "●" and read.hint.isVisibleTo(page) and read.hint.text() == "Reading frames: 200 of 1,000"
    assert tree.row("windows").mark.text() == "○" and not tree.row("joint.fit").isVisibleTo(page)  # folded away
    page._live_tree_chosen("import.centre")
    assert page.live_right.currentIndex() == 1 and page.live_detail_title.text() == "Find the beam centre"
    picture = page.live_detail_picture
    assert picture.kind == "detector" and picture.detail.arrays["image"].shape == (8, 10)
    assert picture.words.text().startswith("Beam centre ≈ (20.0, 26.0) px · detector binned 4 × 4\n40 frames summed · "
                                           "100 of 1,000 pixels left out by the mask")
    page.live_detail_back.click()  # back to the analysis: the details are dropped
    assert page.live_right.currentIndex() == 0 and picture.detail is None and not picture.figure.axes
    page._live_tree_chosen("import.read")  # nothing saved yet: the analysis is followed
    assert page.live_right.currentIndex() == 0
    page._client_event(_step("import.read", "done", "300 of 1000 frames in 75 s", "reading.json", "intake"))
    page._live_tree_chosen("import.read")  # its file is not there yet: said, not a failure
    assert "missing" in picture.words.text()
    page._client_event(_step("import", "done", step="intake"))
    page._live_draw()
    assert not tree.row("import.centre").isVisibleTo(page)  # a done parent folds its children
    tree.row("import").toggle.clicked.emit()
    assert tree.row("import.centre").isVisibleTo(page)
    _close(page)


def test_the_joint_step_follows_its_iterations_and_the_preliminary_result_keeps_the_steps(tmp_path):
    page = _running(tmp_path)
    trace = tmp_path / "out" / "trace"
    write_details(trace)
    for event in (_step("import", "done", step="intake"), _step("joint", "running"), _step("joint.fit", "running"),
                  _event("joint_stage", stage="fit", windows=20, total=6), _progress(1, 5, 50.0)):
        page._client_event(event)
    page._live_draw()
    hint = page.live_tree.row("joint.fit").hint
    assert hint.isVisibleTo(page) and hint.text().startswith("Joint fit: iteration 1 of 6 · derivatives: 5 of 20 windows")
    page._client_event(_step("joint.fit", "running", detail="joint.json"))
    page._live_tree_chosen("joint.fit")
    picture = page.live_detail_picture
    assert picture.kind == "joint" and picture.words.text().startswith("1 of at most 6 iterations")
    write_joint(trace, 2)
    page._client_event(_joint(2, 200.0))  # the open joint fit is read again
    assert picture.words.text().startswith("2 of at most 6 iterations · median fit deviation 11 → 10")
    # the preliminary result: Steps beside its note opens the live view, and back
    out = write_result(tmp_path / "prelim")
    page._client_event(_event("chosen", family="zz_v4", deviations={"zz_v4": 12.5}))
    page._client_event(_event("preliminary", family="zz_v4", folder=str(out)))
    page._live_draw()
    assert page._preliminary_shown() and page.steps_button.isVisibleTo(page)
    page.steps_button.click()
    from src.gimap.features.sputter.presentation.live_panel import LIVE_PAGE

    assert page.result_stack.currentIndex() == LIVE_PAGE and page.live_preliminary_button.isVisibleTo(page)
    page.live_preliminary_button.click()
    assert page.result_stack.currentIndex() == 1 and page._preliminary_shown()
    _close(page)


# -- after the run: Details ▸ Steps ---------------------------------------------------------------------------------------


def test_the_result_keeps_its_steps_for_details(tmp_path):
    page = _running(tmp_path)
    out = write_result(tmp_path / "out")
    write_details(out / "trace")
    write_detector(out / "trace")
    for event in (_step("import.centre", "done", "given: column 12.5, row 30.0", "detector.json", "intake"),
                  _step("import", "done", "300 of 1000 frames in 75 s", step="intake"),
                  _step("compare", "done", "zz_v4 fits better (zz_p5 21.5, zz_v4 12.5)", "compare.json")):
        page._client_event(event)
    page._client_finished(page.service.client.open(out))
    assert (out / "trace" / "gimap_steps.json").is_file()
    page.show_details(True)
    page.tabs.setCurrentIndex(4)
    assert page.tab_steps.currentIndex() == 1 and page.steps_note.text() == "Click a step marked › to see what it found."
    assert page.steps_tree.row("import").summary.full_text() == "300 of 1000 frames in 75 s"
    page._result_step_chosen("compare")
    assert page.steps_picture.kind == "compare" and page.steps_picture.words.text() == "zz_v4 12.5 (chosen)\nzz_p5 21.5"
    page._result_step_chosen("import")  # a parent that saved nothing itself: its first child that did
    assert page.steps_tree.current() == "import.centre" and page.steps_picture.kind == "detector"
    assert page.steps_tree.row("import.centre").isVisibleTo(page)
    assert page.steps_detail_summary.text() == "given: column 12.5, row 30.0"
    page._result_step_chosen("windows")  # no detail recorded for it in this run
    assert page.steps_picture.detail is None and page.steps_picture.words.text() == "This step saved no details."
    for name in ("reading", "onset", "windows", "joint"):  # every kind of detail draws
        page.steps_picture.show(page.service.client.step_detail([out / "trace"], name + ".json"))
        assert page.steps_picture.kind == name and page.steps_picture.figure.axes, name
    words = page.steps_picture.words.text()
    page.steps_picture.show(page.service.client.step_detail([out / "trace"], "reading.json"))
    assert page.steps_picture.words.text().split("\n") == [
        "300 of 1,000 frames read in 1:15 · Processes: 4 · disk 0.2 s per frame (60 % of the time) · 1,500 MB read",
        "only the frames of 12 time windows, every 2 s", "Frames read: 0–99, 400–449, 900–999"]
    assert words.startswith("1 of at most 6 iterations")
    older = write_result(tmp_path / "older")  # an older package: no trace
    page.open_client_result(older)
    page.show_details(True)
    assert page.steps_note.text().startswith("This result has no record of its steps") and not page.steps_tree.shown_ids()
    rebuilt = write_result(tmp_path / "rebuilt")  # a trace but no record of GIMaP's: rebuilt from the files
    write_details(rebuilt / "trace")
    page.open_client_result(rebuilt)
    page.show_details(True)
    assert page.steps_tree.row("windows").title.full_text() == "Lay out the time windows  ›"
    _close(page)


def test_the_steps_speak_chinese(tmp_path):
    from src.gimap.app.presentation.i18n import apply_language
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation import step_pictures, step_texts, storage_panel
    from src.gimap.features.sputter.presentation.step_texts import summary_text

    from src.gimap.features.sputter.presentation.export_flow import EXPORT_TEMPLATES

    for template in (*step_texts.STEP_TEMPLATES, *step_pictures.PICTURE_TEMPLATES, *storage_panel.STORAGE_TEMPLATES,
                     *EXPORT_TEMPLATES):
        assert template in ZH, template
        assert sorted(set(re.findall(r"{(\w+)}", ZH[template]))) == sorted(set(re.findall(r"{(\w+)}", template))), template
    page = _page(tmp_path)
    try:
        apply_language("zh", roots=(page,))
        assert summary_text("found in the frames: column 12.5, row 30.0") == "由帧求得：列 12.5，行 30.0"
        assert summary_text("1.2 s (given)") == "1.2 s（给定）"
        assert summary_text("imported before") == "此前已读入"
        assert summary_text("an invented line the page does not know") == "an invented line the page does not know"
    finally:
        apply_language("en", roots=(page,))
    _close(page)


# -- the continuous metal layer --------------------------------------------------------------------------------------


def _base_windows() -> list:
    return [window(t, [entry("candidate 1", 3.0, 1.0 + t, rank=1, weight=1.0, base_nm=0.5 * t)], pick="candidate 1",
                   smooth="candidate 1") for t in (2.0, 4.0)]


def test_the_continuous_layer_is_shown_when_the_result_lists_it(tmp_path):
    page = _page(tmp_path)
    page.show()
    wanted = WANTED[:2] + [{"key": "base_nm", "label": "thickness of the continuous metal layer under the grains "
                                                       "(0: separate islands)", "unit": "nm"}] + WANTED[2:]
    page.open_client_result(write_result(tmp_path / "new", windows=_base_windows(), wanted=wanted))
    assert _tiles(page)["base_nm"] == "1" and page.tile_captions["base_nm"].full_text() == "Continuous layer t_b"
    assert page.evolution.series["base_nm"][1] == [1.0, 2.0]
    page.show_details(True)
    headers = [page.windows_table.horizontalHeaderItem(c).text() for c in range(page.windows_table.columnCount())]
    assert "Continuous layer t_b (nm)" in headers
    page.open_client_result(write_result(tmp_path / "old", windows=_base_windows()))  # not listed: kept for the picture
    assert "base_nm" not in _tiles(page) and "base_nm" not in page.evolution.series
    headers = [page.windows_table.horizontalHeaderItem(c).text() for c in range(page.windows_table.columnCount())]
    assert "Continuous layer t_b (nm)" not in headers
    assert page.client_result.windows[0].entries[0].value("base_nm") == 1.0
    from src.gimap.app.presentation.i18n import translate

    assert translate("Continuous layer t_b", "zh") == "连续金属层 t_b"
    _close(page)


def test_the_live_view_keeps_only_the_pictures_it_shows():
    run = LiveRun()
    run.feed(_event("windows", family="zz_v4", total=50))
    arrays = {"q": np.linspace(0.1, 1.0, 4), "af": np.linspace(0.1, 0.5, 3), "measured": np.ones((3, 4)),
              "model": np.ones((3, 4))}
    for index in range(50):
        run.feed(LiveEvent("window", "predict", {"family": "zz_v4", "t": float(index), "label": "candidate 1",
                                                 "values": {"R_nm": 1.0}}, dict(arrays)))
    assert len(run.latest) == 1 and run.latest["zz_v4"].number("t") == 49.0  # one picture: the latest
    assert all(not isinstance(value, LiveEvent) for entries in run.solutions["zz_v4"].values()
               for value in entries.values())


# -- Export --------------------------------------------------------------------------------------------------------------


def test_export_writes_the_solutions_shown_with_the_users_choices(tmp_path):
    from tests.sputter_client_fixtures import FakeClientRunner
    from tests.test_sputter_main_screen import _settle

    runner = FakeClientRunner()
    page = _page(tmp_path, client=runner)
    assert not page.export_button.isEnabled() and not page.details_export_button.isEnabled()  # nothing shown
    out = write_result(tmp_path / "out")
    page.open_client_result(out)
    assert page.export_button.isEnabled() and page.details_export_button.isEnabled()
    result = page.client_result
    window = next(w for w in result.windows if len(w.labels()) > 1)
    other = next(label for label in window.labels() if label != window.chosen_label(page.client_choices))
    page.client_choices = page.service.client.with_choice(window, other, page.client_choices)
    page.export_result(tmp_path / "paper")
    assert not page.export_button.isEnabled()  # while it writes
    _settle(lambda: page.export_button.isEnabled())
    folder, target, shown = runner.exports[-1]
    assert folder == out and target.parent == tmp_path / "paper" and target.name.endswith("_export")
    chosen = {item["t"]: item["label"] for item in shown}
    assert chosen[window.time_s] == other and len(chosen) == sum(1 for w in result.windows if w.labels())
    _close(page)


def test_a_package_without_the_export_says_so(tmp_path):
    from src.gimap.features.sputter.application import EXPORT_OLD, ClientFailed
    from src.gimap.features.sputter.infrastructure.client import ClientPackageRunner
    from tests.sputter_client_fixtures import package_folder

    home = package_folder(tmp_path / "package")
    runner = ClientPackageRunner(lambda: tmp_path / "store", chosen=home)
    with pytest.raises(ClientFailed) as caught:
        runner.export(tmp_path / "result", tmp_path / "out", [{"t": 1.0, "label": "joint"}])
    assert str(caught.value) == EXPORT_OLD
