"""Release 0.1.0, Fitting: a running fit shows how far it is and says what it does in the interface language
(R20, R38), and the solutions follow a switch of that language (R8, R42)."""

from __future__ import annotations

import json
import threading
import time
import types

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import LINEAR, PHASES, PROGRESS_LINE, FitData, evaluate, fit
from src.gimap.features.fitting.presentation.single import run
from src.gimap.features.fitting.presentation.single.run import (
    EVALUATED,
    START,
    START_OF,
    _ENGINE_LINE,
    progress_line,
    start_fraction,
)
from tests.test_fit_page import Q_NM, _curve_file, _page, _shape_solution, _truth, _wait

NEW_CHINESE = {
    START: "正在尝试第 {n} / {total} 个初值",
    START_OF: "正在尝试第 {n} / {total} 个初值：{model}",
    EVALUATED: "{line} — 已计算 {evaluations} 次",
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
    assert progress_line({"start": 2, "starts": 6, "types": [3]}) == "Trying start 2 of 6: Vertical cylinder"
    assert progress_line({"start": 1, "starts": 2, "types": [1, 2]}) == "Trying start 1 of 2: Sphere + Random cylinder"
    assert progress_line({"start": 3, "starts": 3}) == "Trying start 3 of 3"  # a fitter that names no family
    working = {"start": 6, "starts": 6, "types": [2], "evaluations": 468, "budget": 1350}
    assert progress_line(working) == "Trying start 6 of 6: Random cylinder — 468 evaluations"
    assert progress_line({**working, "evaluations": 0}) == "Trying start 6 of 6: Random cylinder"


def test_a_start_counts_for_the_share_of_its_budget_it_has_used() -> None:
    assert start_fraction({"start": 1, "starts": 6}) == 0.0  # a fitter that tells only its starts
    assert start_fraction({"start": 4, "starts": 6}) == pytest.approx(0.5)
    assert start_fraction({"start": 6, "starts": 6, "evaluations": 675, "budget": 1350}) == pytest.approx(5.5 / 6)
    assert start_fraction({"start": 6, "starts": 6, "evaluations": 5000, "budget": 1350}) < 1.0  # never done while it runs
    assert start_fraction({"start": 2, "starts": 2, "evaluations": 10, "budget": 0}) < 1.0


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
    family = chinese.ZH["Vertical cylinder"]
    assert progress_line({"start": 2, "starts": 6, "types": [3]}) == chinese.ZH[START_OF].format(n=2, total=6, model=family)
    assert progress_line({"start": 2, "starts": 6}) == chinese.ZH[START].format(n=2, total=6)
    assert progress_line({"start": 2, "starts": 6, "types": [3], "evaluations": 41}) == chinese.ZH[EVALUATED].format(
        line=chinese.ZH[START_OF].format(n=2, total=6, model=family), evaluations=41)


def test_the_quick_fit_numbers_its_starts_in_the_details_and_leaves_the_jobs_count_alone() -> None:
    from src.gimap.features.fitting.application.workflow_v5 import validate_options
    from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import MAX_NFEV, fit_candidates
    from tests.test_experimental_fitting import synthetic_item

    options = validate_options(dict(method="experimental", components=[1], sigma_res=0.008, nu_res=2.1))
    options["distance_hint_nm"] = 3.9  # a third start, from this distance
    seen: list[tuple] = []
    rows = fit_candidates(synthetic_item(), options, lambda *args: seen.append(args), lambda: False)
    assert rows and len(seen) == 3
    # (done, total) count the files of a 1D Predict job (workflow_v5.run_workflow_job): never moved here.
    assert [(done, total) for done, total, _message, _details in seen] == [(0, 1)] * 3
    assert [details for *_rest, details in seen] == [{"start": number, "starts": 3, "types": [1]} for number in (1, 2, 3)]
    assert all("numerical fit [1], initial R=" in message for _d, _t, message, _details in seen)  # the job's log line
    json.dumps([details for *_rest, details in seen], allow_nan=False)  # the job runner sends them between processes

    # A page of one curve may also be told every evaluation; the job is not (its report writes a log line per call).
    told: list[tuple] = []
    beats: list[dict] = []
    watched = fit_candidates(synthetic_item(), options, lambda *args: told.append(args), lambda: False, beats.append)
    assert told == seen  # nothing more is reported
    numbers = ("components", "global_params", "best_chi2_weighted", "best_log_rmse", "nfev", "native_fit")
    assert [[row[key] for key in numbers] for row in watched] == [[row[key] for key in numbers] for row in rows]
    budget = MAX_NFEV * (4 + 1)  # one sphere: R, σR, the distance and its spread are free; the resolution is given
    for number in (1, 2, 3):
        mine = [beat for beat in beats if beat["start"] == number]
        assert [beat["evaluations"] for beat in mine] == list(range(1, len(mine) + 1)) and len(mine) > 5
        assert all(beat == {"start": number, "starts": 3, "types": [1], "evaluations": beat["evaluations"],
                            "budget": budget} for beat in mine)
        assert mine[-1]["evaluations"] <= budget + 1  # an upper bound: the bar never runs past the start's share


def test_the_shape_search_moves_the_bar_and_names_each_start(tmp_path) -> None:
    def quick_fit(_q, _intensity, _sigma, *, report=None, **_options):
        for number, family in enumerate((1, 3, 2), start=1):
            report(0, 1, f"positive: numerical fit [{family}], initial R=1.5 nm",
                   {"start": number, "starts": 3, "types": [family]})
        return [_shape_solution(6.0)]

    page = _page(quick_fit=quick_fit)
    page.open_curve(_curve_file(tmp_path))
    bars = (page.fit_progress, page.status_progress)
    seen: list[tuple] = []
    page._progress.changed.connect(lambda *_args: seen.append((
        [(bar.maximum(), bar.value()) for bar in bars], page.fit_progress_text.text(), page.status_label.text())))
    assert page.method() == "shapes"
    page.run_fit()
    assert all(bar.isVisible() and (bar.minimum(), bar.maximum()) == (0, 0) for bar in bars)  # busy at once, not “0 %”
    _wait(page)
    assert [state for state, _text, _status in seen] == [[(0, 0)] * 2, [(100, 33)] * 2, [(100, 66)] * 2]
    assert [text for _state, text, _status in seen] == [
        "Trying start 1 of 3: Sphere", "Trying start 2 of 3: Vertical cylinder", "Trying start 3 of 3: Random cylinder"]
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


def test_a_long_start_is_seen_to_be_at_work(tmp_path, monkeypatch) -> None:
    """On a real cut the last start (a random cylinder) took nine of twelve seconds: the bar and the line
    must not stand still that long."""
    def quick_fit(_q, _intensity, _sigma, *, report=None, evaluated=None, **_options):
        start = {"start": 2, "starts": 2, "types": [2]}
        report(0, 1, "positive: numerical fit [2], initial R=4 nm", dict(start))
        for evaluations in (300, 600, 5000):  # the last: past its budget (cannot be, but never “done” while it runs)
            evaluated({**start, "evaluations": evaluations, "budget": 1200})
        return [_shape_solution(6.0)]

    monkeypatch.setattr(run, "HEARTBEAT", 0.0)  # every evaluation told here is said
    page = _page(quick_fit=quick_fit)
    page.open_curve(_curve_file(tmp_path))
    seen = _watch(page)
    page.run_fit()
    _wait(page)
    name = "Trying start 2 of 2: Random cylinder"
    assert seen == [(100, 50, 50, name, name)] + [
        (100, percent, percent, f"{name} — {evaluations} evaluations", f"{name} — {evaluations} evaluations")
        for percent, evaluations in ((62, 300), (75, 600), (99, 5000))]
    assert page.session.solutions and not page.fit_progress.isVisible()

    del seen[:]
    monkeypatch.setattr(run, "HEARTBEAT", 3600.0)  # … but not more often than this: only the start itself
    page.method_buttons["shapes"].setChecked(True)
    page.run_fit()
    _wait(page)
    assert seen == [(100, 50, 50, name, name)]
    page.dispose()


def test_the_real_shape_search_tells_its_evaluations_through_the_page(tmp_path, monkeypatch) -> None:
    from src.gimap.features.fitting.bootstrap import create_quick_fit

    monkeypatch.setattr(run, "HEARTBEAT", 0.0)
    page = _page(quick_fit=create_quick_fit())
    page.open_curve(_curve_file(tmp_path))
    page.families_combo.setCurrentIndex(max(0, page.families_combo.findData("model")))  # the model's family: two starts
    seen = _watch(page)
    page.run_fit()
    _wait(page, 120.0)
    assert page.session.solutions, page.status_label.text()
    counted = [value for maximum, value, *_rest in seen if maximum == 100]
    assert counted == sorted(counted) and counted[-1] < 100  # forwards only, through both starts
    lines = [line for *_bars, line, _status in seen]
    assert lines[0] == "Trying start 1 of 2: Sphere" and "Trying start 2 of 2: Sphere" in lines
    assert sum(1 for line in lines if line.startswith("Trying start 1 of 2: Sphere — ")) > 5
    assert all(line == status for *_bars, line, status in seen)
    page.dispose()


def test_a_fit_of_the_engine_counts_on_the_bar(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_model(page.session.model.with_values({(0, "R"): 5.3, (0, "D"): 44.0}))
    page.method_buttons["global"].setChecked(True)
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

    def quick_fit(_q, _intensity, _sigma, *, report=None, **_options):
        report(0, 1, "positive: numerical fit [3], initial R=4 nm", {"start": 2, "starts": 6, "types": [3]})
        reported.set()
        release.wait(60)
        return [_shape_solution(6.0)]

    page = _page(quick_fit=quick_fit)
    try:
        page.open_curve(_curve_file(tmp_path))
        page.run_fit()
        end = time.monotonic() + 30
        while not page.fit_progress_text.text() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        assert reported.is_set() and page.fit_running()
        english = "Trying start 2 of 6: Vertical cylinder"
        assert page.fit_progress_text.text() == english == page.status_label.text()
        assert (page.fit_progress.maximum(), page.fit_progress.value()) == (100, 16)
        _switch(chinese, page, "zh")
        said = chinese.ZH[START_OF].format(n=2, total=6, model=chinese.ZH["Vertical cylinder"])
        assert page.fit_progress_text.text() == said == page.status_label.text()
        assert (page.fit_progress.maximum(), page.fit_progress.value()) == (100, 16)
        _switch(chinese, page, "en")
        assert page.fit_progress_text.text() == english == page.status_label.text()
    finally:
        release.set()
        _wait(page)
        page.dispose()


def _until_said(page, limit: float = 30.0) -> None:
    end = time.monotonic() + limit
    while not page.fit_progress_text.text() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_stop_during_the_shape_search_is_said_as_a_stop_not_as_a_failure(tmp_path) -> None:
    def quick_fit(_q, _intensity, _sigma, *, report=None, cancelled=None, **_options):
        report(0, 1, "positive: numerical fit [1], initial R=1.5 nm", {"start": 1, "starts": 6, "types": [1]})
        end = time.monotonic() + 60
        while not cancelled() and time.monotonic() < end:
            time.sleep(0.01)
        raise RuntimeError("Fitting cancelled")  # as the quick fit does (experimental_fit.fit_candidates)

    page = _page(quick_fit=quick_fit)
    page.open_curve(_curve_file(tmp_path))
    model = page.session.model
    page.run_fit()
    _until_said(page)
    page.stop_fit()
    assert page.fit_progress_text.text() == "Stopping…"  # nothing is kept of a search cut short
    page._progress.changed.emit(0.5, {"start": 4, "starts": 6, "types": [3]})  # a line sent just before Stop
    assert page.fit_progress_text.text() == "Stopping…" and page.fit_progress.maximum() == 0  # … is not said after it
    _wait(page)
    assert page.status_label.text() == "Stopped before the search finished: no new solutions."
    assert "failed" not in page.status_label.text() and "cancelled" not in page.status_label.text()
    assert page.session.solutions == [] and page.session.model == model and page.step_rail.state("fit") == "pending"

    def broken(*_args, **_options):
        raise RuntimeError("the fitter broke")

    page.quick_fit = broken  # a failure nobody asked for is still said as one
    page.run_fit()
    _wait(page)
    assert page.status_label.text() == "The fit failed: the fitter broke"
    page.method_buttons["global"].setChecked(True)  # a fit of the engine keeps its best values when stopped
    page.run_fit()
    _until_said(page)
    page.stop_fit()
    assert page.fit_progress_text.text() == "Stopping … the best values so far are kept."
    _wait(page)
    assert page.status_label.text().startswith("Stopped: the best values so far are kept (")
    page.dispose()


def test_the_solutions_and_what_is_said_of_them_follow_the_language(tmp_path, chinese) -> None:
    page = _page(quick_fit=lambda *_args, **_options: [_shape_solution(radius) for radius in (5.0, 6.0, 7.0)])
    page.open_curve(_curve_file(tmp_path))
    page.run_fit()
    _wait(page)
    table = page.solutions_table

    def names() -> list[str]:
        return [table.item(line, 1).text() for line in range(table.rowCount())]

    assert names() == ["Sphere"] * 3 and page.step_rail.detail("fit").startswith("3 solutions · best Sphere ")
    assert page.status_label.text().startswith("3 solutions; the best, Sphere (")
    sphere = chinese.ZH["Sphere"]
    _switch(chinese, page, "zh")
    assert names() == [sphere] * 3  # composed again, not the text of when they were found
    assert sphere in page.step_rail.detail("fit") and "Sphere" not in page.step_rail.detail("fit")
    assert sphere in page.status_label.text() and "Sphere" not in page.status_label.text()
    table.selectRow(1)
    page.use_selected_solution()
    chosen = "{model} is in Model (Undo brings the previous one back)."
    assert page.status_label.text() == chinese.ZH[chosen].format(model=sphere)
    _switch(chinese, page, "en")
    assert names() == ["Sphere"] * 3 and page.status_label.text() == chosen.format(model="Sphere")
    assert page.step_rail.detail("fit").startswith("3 solutions · best Sphere ")
    page.dispose()


def test_a_solution_of_analyze_is_named_in_the_language_of_now(tmp_path, chinese) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    row = dict(_shape_solution(6.0), best_chi2_weighted=1.1)
    assert page.show_solution(row)
    assert page.status_label.text() == "Sphere is in Model; this model draws the same curve."
    _switch(chinese, page, "zh")
    sphere = chinese.ZH["Sphere"]
    assert page.status_label.text() == chinese.ZH["{model} is in Model; this model draws the same curve."].format(model=sphere)
    assert page.solutions_table.item(0, 1).text() == sphere
    _switch(chinese, page, "en")
    assert page.solutions_table.item(0, 1).text() == "Sphere"
    page.dispose()
