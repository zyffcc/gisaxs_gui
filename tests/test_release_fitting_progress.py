"""Release 0.1.0, Fitting: a running fit shows how far it is and says what it does in the interface language
(R20, R38), and the solutions follow a switch of that language (R8, R42). Predict parameters' job is a stand-in
(``tests.test_fit_page._predicting``) except where the real one is said to run."""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import LINEAR, PHASES, PROGRESS_LINE, FitData, evaluate, fit
from src.gimap.features.fitting.presentation.single.run import _ENGINE_LINE, progress_line
from tests.test_fit_page import Q_NM, _analyze_solution, _curve_file, _page, _predicting, _solutions, _truth, _wait

LOADING = "Loading the proposal network"
WEIGHTING = "Weighting the network's samples and refining each solution"
"""The lines of Predict parameters' job (``posterior_job.run_predict``)."""
NEW_CHINESE = {
    PROGRESS_LINE: "{phase} — 已计算 {evaluations} 次，最佳 χ²ᵣ {chi2}",
    "Solving the scales": "正在求解比例",
    "Searching the ranges": "正在范围内搜索",
    "Refining": "正在精修",
}
"""The entries this release adds to the Chinese table (put in here until they are in it)."""


@pytest.fixture
def chinese(monkeypatch):
    """The interface-language module with the new entries; English again afterwards."""
    from src.gimap.app.presentation import i18n

    for english, text in NEW_CHINESE.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, text)
            monkeypatch.setitem(i18n._TO_ENGLISH, text, english)
    yield i18n
    i18n.apply_language("en")


def _switch(i18n, page, language: str) -> None:
    """As the main window does on View ▸ Language: the walker, then the page's own texts."""
    i18n.apply_language(language, [page])
    page.refresh_language()


def test_the_template_of_the_engines_line_is_the_one_the_page_knows() -> None:
    assert PROGRESS_LINE == "{phase} — {evaluations} evaluations, best χ²ᵣ {chi2}"
    assert PHASES == ("Solving the scales", "Searching the ranges", "Refining")
    found = _ENGINE_LINE.fullmatch("Searching the ranges — 318 evaluations, best χ²ᵣ 1.284")
    assert found.groupdict() == {"phase": "Searching the ranges", "evaluations": "318", "chi2": "1.284"}
    assert progress_line("Refining — 1 evaluations, best χ²ᵣ 1.29") == "Refining — 1 evaluations, best χ²ᵣ 1.29"
    assert progress_line("1/1 files finished — Current curve") == "1/1 files finished — Current curve"  # another job's
    assert progress_line(LOADING) == LOADING  # Predict parameters' lines, as they come


def test_every_line_of_a_real_fit_is_said_in_chinese_with_its_numbers_kept(chinese, monkeypatch) -> None:
    from src.gimap.features.fitting.domain import fit_engine

    ticks = iter(range(1, 10_000_000))
    monkeypatch.setattr(fit_engine, "time", types.SimpleNamespace(perf_counter=lambda: float(next(ticks))))  # every evaluation reports
    exact = evaluate(_truth(), Q_NM)
    data = FitData.prepare(Q_NM, exact, np.sqrt(exact) + 0.5)
    start = _truth().with_values({(0, "R"): 5.3, (0, "D"): 44.0})
    scales_only = start
    for path, parameter in start.parameters():
        if parameter.free and path[1] not in LINEAR:
            scales_only = scales_only.with_parameter(path, free=False)
    lines: list[str] = []
    fit(start, data, method="global", max_evaluations=200, progress=lambda _fraction, line: lines.append(line))
    fit(scales_only, data, progress=lambda _fraction, line: lines.append(line))
    found = [_ENGINE_LINE.fullmatch(line) for line in lines]
    assert lines and all(found)  # the engine makes its lines from the template
    assert {item["phase"] for item in found} == set(PHASES)
    chinese.translator().language = "zh"
    for line, item in zip(lines, found):
        said = progress_line(line)
        assert said == chinese.ZH[PROGRESS_LINE].format(
            phase=chinese.ZH[item["phase"]], evaluations=item["evaluations"], chi2=item["chi2"])
        assert "evaluations" not in said and item["phase"] not in said
    assert progress_line(LOADING) == chinese.ZH[LOADING] and progress_line(WEIGHTING) == chinese.ZH[WEIGHTING]


def test_predict_parameters_moves_the_bar_and_says_each_step(tmp_path) -> None:
    def answer(payload, report, _cancelled):
        for done, total, line in ((0, 1, LOADING), (1, 3, WEIGHTING), (2, 3, "Refining solution 2 of 3")):
            report(done, total, line)
        return _solutions(payload, (6.0,))

    page = _page()
    _predicting(page, answer)
    page.open_curve(_curve_file(tmp_path))
    bars = (page.fit_progress, page.status_progress)
    seen: list[tuple] = []
    page._progress.changed.connect(lambda *_args: seen.append((
        [(bar.maximum(), bar.value()) for bar in bars], page.fit_progress_text.text(), page.status_label.text())))
    page.show_step("fit")
    page.set_method("predict")
    page.run_fit()
    assert all(bar.isVisible() and (bar.minimum(), bar.maximum()) == (0, 0) for bar in bars)  # busy at once, not “0 %”
    _wait(page)
    assert [state for state, _text, _status in seen] == [[(0, 0)] * 2, [(100, 33)] * 2, [(100, 66)] * 2]
    assert [text for _state, text, _status in seen] == [LOADING, WEIGHTING, "Refining solution 2 of 3"]
    assert all(text == status for _state, text, status in seen)  # the status line says the same
    assert all((bar.maximum(), bar.value()) == (100, 0) and not bar.isVisible() for bar in bars)  # ready for the next fit
    assert page.fit_progress_text.text() == "" and page.session.solutions
    page.dispose()


def _watch(page) -> list[tuple]:
    seen: list[tuple] = []
    page._progress.changed.connect(lambda *_args: seen.append(
        (page.fit_progress.maximum(), page.fit_progress.value(), page.status_progress.value(),
         page.fit_progress_text.text(), page.status_label.text())))
    return seen


def test_the_real_predict_job_tells_its_steps_through_the_page(tmp_path, monkeypatch) -> None:
    """The job in its own process (with the stored random network of tests/data/ml1d)."""
    from tests.test_fit_page_predict import NETWORK, _sphere_file
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(NETWORK))
    page = _page()
    try:
        assert page.open_curve(_sphere_file(tmp_path))
        page.set_model(FitModel((new_component("sphere", radius=3.0),)))
        page.set_method("predict")
        seen = _watch(page)
        page.run_fit()
        _wait(page, 300.0)
        assert page.session.solutions, page.status_label.text()
        lines = [line for *_bars, line, _status in seen]
        assert lines[0].startswith(LOADING + " · 0:0")  # its stages (predict_progress.stage_line), with the time
        assert any(line.startswith("Step 1 of 3 · Network proposals: composition 1 of 1 · ") for line in lines)
        assert any(line.startswith("Step 2 of 3 · Refining the modes: composition 1 of 1") for line in lines)
        assert any(line.startswith("Step 3 of 3 · Sampling the posterior: solution 1 of ") for line in lines)
        assert all(line == status for *_bars, line, status in seen)
        assert not page.fit_progress.isVisible() and page.fit_progress_text.text() == ""
    finally:
        page.dispose()


def test_a_fit_of_the_engine_counts_on_the_bar(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_model(page.session.model.with_values({(0, "R"): 5.3, (0, "D"): 44.0}))
    page.set_method("global")
    seen: list[tuple] = []
    page._progress.changed.connect(lambda *_args: seen.append(
        (page.fit_progress.maximum(), page.fit_progress.value(), page.fit_progress_text.text())))
    page.run_fit()
    assert page.fit_progress.maximum() == 0
    _wait(page)
    assert seen and all(_ENGINE_LINE.fullmatch(text) for _maximum, _value, text in seen)
    counted = [value for maximum, value, _text in seen if maximum == 100]
    assert counted and counted == sorted(counted) and 0 < counted[-1] <= 99  # it moves, and only forwards
    assert all(value == 0 for maximum, value, _text in seen if maximum == 0)  # busy only while nothing is counted
    page.dispose()


def test_a_running_fit_says_its_line_again_after_a_switch_of_the_language(tmp_path, chinese) -> None:
    reported, release = threading.Event(), threading.Event()

    def answer(payload, report, _cancelled):
        report(1, 6, WEIGHTING)
        reported.set()
        release.wait(60)
        return _solutions(payload, (6.0,))

    page = _page()
    _predicting(page, answer)
    try:
        page.open_curve(_curve_file(tmp_path))
        page.set_method("predict")
        page.run_fit()
        end = time.monotonic() + 30
        while not page.fit_progress_text.text() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        assert reported.is_set() and page.fit_running()
        assert page.fit_progress_text.text() == WEIGHTING == page.status_label.text()
        assert (page.fit_progress.maximum(), page.fit_progress.value()) == (100, 16)
        _switch(chinese, page, "zh")
        said = chinese.ZH[WEIGHTING]
        assert page.fit_progress_text.text() == said == page.status_label.text()
        assert (page.fit_progress.maximum(), page.fit_progress.value()) == (100, 16)
        _switch(chinese, page, "en")
        assert page.fit_progress_text.text() == WEIGHTING == page.status_label.text()
    finally:
        release.set()
        _wait(page)
        page.dispose()


def _until_said(page, limit: float = 30.0) -> None:
    end = time.monotonic() + limit
    while not page.fit_progress_text.text() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_stop_during_predict_parameters_is_said_as_a_stop_not_as_a_failure(tmp_path) -> None:
    def answer(_payload, report, cancelled):
        report(0, 1, LOADING)
        end = time.monotonic() + 60
        while not cancelled() and time.monotonic() < end:
            time.sleep(0.01)
        raise RuntimeError("Fitting cancelled")  # as the job ends when the runner cancels it

    page = _page()
    jobs = _predicting(page, answer)
    page.open_curve(_curve_file(tmp_path))
    model = page.session.model
    page.set_method("predict")
    page.run_fit()
    _until_said(page)
    page.stop_fit()
    assert page.fit_progress_text.text() == "Stopping…"  # nothing is kept of a search cut short
    page._progress.changed.emit(0.5, WEIGHTING)  # a line sent just before Stop
    assert page.fit_progress_text.text() == "Stopping…" and page.fit_progress.maximum() == 0  # … is not said after it
    _wait(page)
    assert jobs.cancelled  # the job was asked to stop
    assert page.status_label.text() == "Stopped before any solution was finished: no new solutions."
    assert "failed" not in page.status_label.text() and "cancelled" not in page.status_label.text()
    assert page.session.solutions == [] and page.session.model == model and page.step_rail.state("fit") == "pending"

    def broken(*_args):
        raise RuntimeError("the fitter broke")

    jobs.answer = broken  # a failure nobody asked for is still said as one
    page.run_fit()
    _wait(page)
    assert page.status_label.text() == "The fit failed: the fitter broke"
    page.set_method("global")  # a fit of the engine keeps its best values when stopped
    page.run_fit()
    _until_said(page)
    page.stop_fit()
    assert page.fit_progress_text.text() == "Stopping … the best values so far are kept."
    _wait(page)
    assert page.status_label.text().startswith("Stopped: the best values so far are kept (")
    page.dispose()


def test_the_solutions_and_what_is_said_of_them_follow_the_language(tmp_path, chinese) -> None:
    page = _page()
    _predicting(page, (6.0, 5.0, 7.0))
    page.open_curve(_curve_file(tmp_path))
    page.set_method("predict")
    page.run_fit()
    _wait(page)
    table = page.solutions_table

    def names() -> list[str]:
        return [table.item(line, 1).text() for line in range(table.rowCount())]

    numbered = [f"Sphere #{number}" for number in (1, 2, 3)]  # one composition: told apart by their number
    assert names() == numbered and page.step_rail.detail("fit").startswith("3 solutions · best Sphere #1 ")
    assert page.status_label.text().startswith("3 solutions; the best, Sphere #1 (")
    sphere = chinese.ZH["Sphere"]
    _switch(chinese, page, "zh")
    assert names() == [f"{sphere} #{number}" for number in (1, 2, 3)]  # composed again, not the text of when found
    assert sphere in page.step_rail.detail("fit") and "Sphere" not in page.step_rail.detail("fit")
    assert sphere in page.status_label.text() and "Sphere" not in page.status_label.text()
    table.selectRow(1)
    page.use_selected_solution()
    chosen = "{model} is in Model (Undo brings the previous one back)."
    assert page.status_label.text() == chinese.ZH[chosen].format(model=f"{sphere} #2")
    _switch(chinese, page, "en")
    assert names() == numbered and page.status_label.text() == chosen.format(model="Sphere #2")
    assert page.step_rail.detail("fit").startswith("3 solutions · best Sphere #1 ")
    page.dispose()


def test_a_solution_of_analyze_is_named_in_the_language_of_now(tmp_path, chinese) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    row = dict(_analyze_solution(6.0), best_chi2_weighted=1.1)
    assert page.show_solution(row)
    assert page.status_label.text() == "Sphere is in Model; this model draws the same curve."
    _switch(chinese, page, "zh")
    sphere = chinese.ZH["Sphere"]
    assert page.status_label.text() == chinese.ZH["{model} is in Model; this model draws the same curve."].format(model=sphere)
    assert page.solutions_table.item(0, 1).text() == sphere
    _switch(chinese, page, "en")
    assert page.solutions_table.item(0, 1).text() == "Sphere"
    page.dispose()
