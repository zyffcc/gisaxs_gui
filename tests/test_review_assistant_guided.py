"""Review fixes of the automatic analysis (area 5): frames of a series, Clear, step lines, words, cut rows, dock.

Each test names the finding it guards: results of a series stay tied to the frames the run analysed,
Clear forgets the reports and the answers, a failed step keeps its reason, the run's texts are
translatable, the Results tab counts the cut's rows as Analyze does, the AI dock's title follows a
language switch, and Auto hands its detected technique to the run instead of switching the mode.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QPushButton

from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
from src.gimap.features.assistant.presentation import GuidedAnalysis
from src.gimap.features.assistant.presentation.guided_results import GuidedResultsPanel
from src.gimap.features.assistant.presentation.guided_text import (
    detected_technique,
    other_frames,
    step_summary,
)
from tests.test_assistant_gui import _app, _wait, analyze  # noqa: F401 - the fixture
from tests.test_guided_gisaxs import galaxi_run  # noqa: F401 - the fixture


def _series(path: Path, frames: int = 12) -> Path:
    import h5py

    data = np.random.default_rng(0).poisson(5, (frames, 64, 64)).astype(np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/instrument/detector/data", data=data)
    return path


def _report(frame, *, frames=None, attention=()) -> dict:
    report = {"ok": True, "procedure": "giwaxs", "frame": str(frame), "peaks": [], "rings": [],
              "needs_attention": list(attention), "decisions": [], "steps": []}
    if frames is not None:
        report["frames"] = frames
    return report


def _actions(guided) -> dict:
    save = guided.results.findChild(QPushButton, "guidedSaveReport")
    compare = guided.results.findChild(QPushButton, "guidedCompareButton")
    return {"save": save.isEnabled(), "compare": compare.isEnabled() if compare is not None else None,
            "card": guided.progress_panel.save_button.isEnabled()}


def _show(page, frame_index: int, summed: int) -> None:
    page.view_model.set_sum_count(summed)
    page.view_model.set_frame(frame_index)
    page.run_analysis()
    assert page.tasks.wait(60)
    QApplication.processEvents()


# -- results of a series belong to the frames the run analysed ------------------------------------------


def test_results_of_a_series_are_off_while_analyze_shows_other_frames_of_it(analyze, tmp_path: Path) -> None:  # noqa: F811
    _window, page, _context = analyze
    GuidedResultsPanel.details_open = False
    series = _series(tmp_path / "insitu_series.nxs")
    film = Path(page.view_model.current_path)
    guided = GuidedAnalysis(page.automation, save_text=lambda path, text: path)
    page.analysisShown.connect(lambda analysis: guided.frame_shown(str(analysis.path)))
    sent = []
    guided.refineRequested.connect(lambda: sent.append("refine"))
    page.add_paths([series])
    assert page.tasks.wait(60)
    _show(page, 2, 10)  # frames 3–12 summed: what the standard procedure analyses of a series of 12
    report = _report(series, frames={"total": 12, "first": 3, "summed": 10})
    guided._finished(report)
    assert guided._for_this_frame() and _actions(guided) == {"save": True, "compare": True, "card": True}

    _show(page, 0, 1)  # the slider to frame 1: the results stay, nothing acts on them
    assert guided.report is report and not guided.results.isHidden()
    assert _actions(guided) == {"save": False, "compare": False, "card": False}
    assert guided.status_label.text() == "Results are for frames 3–12; run again for this frame"
    assert guided.save_report() is None
    guided.results.refineRequested.emit()
    assert sent == []

    _show(page, 2, 10)  # the run's frames again
    assert _actions(guided) == {"save": True, "compare": True, "card": True}
    assert guided.status_label.text().startswith("Done — see the Results tab.")

    # Another file and back: Analyze starts the series at frame 1 again, so the report comes back switched off.
    rows = {page.file_list.item(row).text(): row for row in range(page.file_list.count())}
    page.file_list.setCurrentRow(next(row for text, row in rows.items() if film.name in text))
    assert page.tasks.wait(60)
    QApplication.processEvents()
    assert guided.report is None and guided.results.isHidden()
    page.file_list.setCurrentRow(next(row for text, row in rows.items() if series.name in text))
    assert page.tasks.wait(60)
    QApplication.processEvents()
    assert guided.report is report and not guided.results.isHidden()
    analysis = page.view_model.state.analysis
    same_frames = (analysis.frame_index + 1, analysis.frame_total) == (3, 10)
    assert _actions(guided)["save"] is same_frames
    if not same_frames:
        assert guided.status_label.text() == "Results are for frames 3–12; run again for this frame"
    guided.results.deleteLater()


def test_the_frames_a_caller_names_count_when_analyze_cannot_say() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    path = "C:/data/insitu_00001.nxs"
    guided.frame_shown(path, frame=394, summed=10)
    guided._finished(_report(path, frames={"total": 403, "first": 394, "summed": 10}))
    assert guided._for_this_frame()
    guided.frame_shown(path, frame=1, summed=10)
    assert not guided._for_this_frame()
    assert guided.status_label.text() == "Results are for frames 394–403; run again for this frame"
    guided.frame_shown(path)  # nothing known about the frames: only the file counts
    assert guided._for_this_frame()
    assert other_frames({"frames": {"total": 1, "first": 1, "summed": 1}}, (5, 1)) is None  # no series
    assert other_frames({"frames": {"total": 20, "first": 11, "summed": 10}}, (11, 10)) is None
    assert other_frames({"frames": {"total": 20, "first": 11, "summed": 10}}, (11, 1)) == (11, 20)
    guided.results.deleteLater()


# -- Clear (and a project opened) forgets the reports and the answers -------------------------------------


def test_clear_forgets_the_reports_and_the_answers() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    frame = "C:/data/film_a.tif"
    question = {"item": "incidence angle αi", "why": "No αi anywhere.", "option": "incidence_deg", "hint": ""}
    report = _report(frame, attention=[question])
    guided.frame_shown(frame)
    guided._finished(report)
    guided.question_fields["incidence_deg"].setText("0.3")
    assert guided.options().incidence_deg == 0.3
    guided.files_cleared()
    assert guided.report is None and guided.results.isHidden() and guided.questions.isHidden()
    assert not guided.progress_panel.save_button.isEnabled() and guided.save_report() is None
    assert guided.status_label.text() == "No AI needed: the standard procedure, each decision with its reason."
    assert guided.options().incidence_deg is None  # a project's own αi is not overridden by an old answer
    guided.frame_shown(frame)  # the same file opened again: its old report does not come back
    assert guided.report is None and guided.results.isHidden()
    guided.results.deleteLater()


def test_clear_during_a_run_stops_it_and_forgets_its_report_when_it_ends() -> None:
    _app()
    guided = GuidedAnalysis(lambda: None)
    frame = "C:/data/film_a.tif"
    guided.frame_shown(frame)
    release = threading.Event()
    guided._thread = threading.Thread(target=release.wait, daemon=True)
    guided._thread.start()
    guided._stop_event = threading.Event()
    stopping = []
    guided.stopping.connect(lambda: stopping.append(True))
    guided.files_cleared()
    assert guided._stop_event.is_set() and stopping == [True]
    release.set()
    guided._thread.join()
    guided._finished(_report(frame))  # the stopped run's result arrives after Clear
    assert guided.report is None and guided.results.isHidden() and not guided._reports
    guided.frame_shown(frame)
    assert guided.report is None
    guided.results.deleteLater()


# -- a failed step keeps its reason -------------------------------------------------------------------


def test_a_failed_set_halves_keeps_its_error() -> None:
    _app()
    assert step_summary("set_halves", {"side": "mean"}, "halves: mean", "en") == "Mean of both halves"
    error = "failed: The analysis failed."
    assert step_summary("set_halves", {"side": "mean"}, error, "en") == error
    assert step_summary("set_halves", {"side": "x"}, "invalid input: side must be one of …", "zh").startswith("invalid input")
    guided = GuidedAnalysis(lambda: None)
    guided._stepped({"state": "start", "tool": "set_halves", "arguments": {"side": "mean"}})
    guided._progress(f"set_halves: {error}")
    assert guided.status_label.text().endswith("— " + error)
    guided.results.deleteLater()


# -- the run's texts are translatable -------------------------------------------------------------------


def test_the_run_texts_go_through_the_table(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    _app()
    table = {
        "Working… (finding a calibration can take a minute)": "工作中…",
        "The analysis stopped: {message}": "分析停止：{message}",
        "Found in the notes: {found}": "笔记中找到：{found}",
        "energy = {value} keV": "能量 = {value} keV",
        "Your earlier answers are kept: {answers}": "保留之前的回答：{answers}",
        "X-ray energy (keV)": "X 射线能量 (keV)",
        "Start versus end: {summary}.": "开始与结束对比：{summary}。",
        "{n} appeared": "出现 {n} 条",
    }
    for english, chinese in table.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)
    guided = GuidedAnalysis(lambda: None)  # no Analyze: the run ends at once with "no image is open"
    started = []
    guided.started.connect(started.append)
    apply_language("zh")
    try:
        guided.run()
        assert started == ["工作中…"] and guided.status_label.text() == "工作中…"
        assert guided.progress_panel.now_label.text() == "工作中…"
        _wait(lambda: not guided._busy(), 60)
        guided._failed("no frame")
        assert guided.status_label.text() == "分析停止：no frame"
        guided.notes_edit.setPlainText("12.4 keV")
        assert guided.notes_found.text() == "笔记中找到：能量 = 12.4 keV"
        guided._answers = {"energy_kev": "12.4"}
        guided._show_questions([])
        assert guided.answers_label.text() == "保留之前的回答：X 射线能量 (keV): 12.4"
        guided.report = _report("C:/data/a.nxs", frames={"total": 20, "first": 11, "summed": 10})
        start = dict(guided.report, peaks=[])
        end = dict(guided.report, peaks=[{"q": 1.0, "d_A": 6.283, "fwhm": 0.01, "area": 1.0, "caveat": ""}])
        guided.report = end
        guided._thread = None
        guided._compared(start)
        assert guided.status_label.text() == "开始与结束对比：出现 1 条。"
    finally:
        apply_language(DEFAULT_LANGUAGE)
    guided.results.deleteLater()


# -- the cut's rows as Analyze counts them --------------------------------------------------------------


def test_the_results_tab_names_the_rows_of_the_cut_as_analyze_does(galaxi_run) -> None:  # noqa: F811
    from src.gimap.features.analyze.presentation.bindings.display import cut_lines
    from src.gimap.features.assistant.presentation.guided_gisaxs import GisaxsResults, cut_rows, pixel_span

    _app()
    session, report = galaxi_run
    analysis = session.page.view_model.state.analysis
    card = next(line for line in cut_lines(analysis) if line.startswith("Horizontal cut"))
    analyze_rows = re.search(r"rows (\d+–\d+)", card).group(1)
    start, stop = analysis.reduction.curve("horizontal").region["rows"]
    assert analyze_rows == f"{start}–{stop - 1}" and cut_rows(report) == analyze_rows
    section = GisaxsResults(report)
    texts = [label.text() for label in section.findChildren(QLabel)]
    assert any(f"rows {analyze_rows}." in text for text in texts), texts
    section.dispose()
    section.deleteLater()
    assert pixel_span((605.0, 610.0)) == "605–609"
    assert pixel_span((-2.5, 3.5)) == "0–3" and pixel_span((1040.2, 1046.0), 1043) == "1040–1042"
    # Analyze's status names [first, last], both included (horizontal_pixel_rows).
    assert pixel_span(None, pixels=(10, 14)) == "10–14" and pixel_span((None, None)) == "?"


# -- the AI dock's title follows a language switch ------------------------------------------------------


def test_the_ai_dock_title_follows_a_later_language_switch(analyze, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    from src.gimap.app.presentation import i18n
    from tests.test_assistant_gui import _controller, _services

    window, page, context = analyze
    monkeypatch.setitem(i18n.ZH, "AI Assistant", "AI 助手")
    controller = _controller(window, page, context, _services(tmp_path, None))
    controller.show_panel()
    QApplication.processEvents()  # the window's new layout, while its plots are alive (not after the fixture closes it)
    dock = window.findChild(QDockWidget, "assistantDock")
    assert dock.windowTitle() == "AI Assistant" and not dock.isWindow()
    try:
        apply_language("zh", [window])  # as Settings ▸ Language does: the main window and everything in it
        assert dock.windowTitle() == "AI 助手"
        apply_language(DEFAULT_LANGUAGE, [window])
        assert dock.windowTitle() == "AI Assistant"
    finally:
        apply_language(DEFAULT_LANGUAGE)
        controller.shutdown()
        QApplication.processEvents()


# -- Auto: the run analyses what Analyze detected, without pinning a mode ---------------------------------


def test_auto_hands_its_detected_technique_to_the_run(analyze) -> None:  # noqa: F811
    _window, page, _context = analyze
    assert page.view_model.state.mode == "auto" and page.view_model.state.analysis.kind == "giwaxs"
    guided = GuidedAnalysis(page.automation)
    assert guided.options().technique == "giwaxs"
    guided._answers = {"calibration": "C:/beamtime/agbh.tif"}  # a new calibration may classify it otherwise
    assert guided.options().technique is None
    guided._answers = {}
    assert detected_technique({"mode": "auto", "measurement": "gisaxs"}, False) == "gisaxs"
    assert detected_technique({"mode": "giwaxs", "measurement": "giwaxs"}, False) is None  # the procedure follows it
    assert detected_technique({"mode": "auto", "measurement": None}, False) is None  # not classified yet
    guided.results.deleteLater()


def test_a_gisaxs_frame_on_auto_gets_the_gisaxs_procedure_and_auto_stays(tmp_path: Path) -> None:
    from src.gimap.app import AppContext
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    _app()
    shape = (300, 300)
    # 1.73 m away, 172 µm pixels, beam near the bottom: a small-angle (GISAXS) set-up for Auto.
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 150.0, 250.0, 1.34, 0.4)
    rows, columns = np.mgrid[0:shape[0], 0:shape[1]]
    image = 5.0 + 4000.0 / (1.0 + ((columns - 150.0) / 6.0) ** 2) * np.exp(-np.abs(rows - 200.0) / 25.0)
    frame = tmp_path / "film_gisaxs.tif"
    Image.fromarray(np.random.default_rng(3).poisson(image).astype(np.float32), mode="F").save(frame)
    context = AppContext(
        settings=InMemorySettingsRepository({}), session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=InMemoryInstrumentProfileRepository([InstrumentProfile("SAXS", geometry, None, shape)]),
    )
    page = AnalyzePage(create_analyze_view_model(context))
    window = QMainWindow()
    window.setCentralWidget(page)
    page.add_paths([frame])
    assert page.tasks.wait(60)
    analysis = page.view_model.state.analysis
    if page.view_model.state.mode != "auto" or analysis.kind != "gisaxs":
        pytest.skip(f"Auto does not read this synthetic frame as GISAXS ({analysis.kind})")
    guided = GuidedAnalysis(page.automation, parent=page)
    page.analysisShown.connect(lambda shown: guided.frame_shown(str(shown.path)))
    guided.run()
    _wait(lambda: guided.report is not None and not guided._busy(), 180)
    report = guided.report
    assert report["procedure"] == "gisaxs", [(step["tool"], step["summary"]) for step in report["steps"]]
    assert not any(step["tool"] == "set_measurement_mode" for step in report["steps"])
    assert page.view_model.state.mode == "auto" and page.mode_combo.currentData() == "auto"
    page.tasks.wait(60)
    window.close()
