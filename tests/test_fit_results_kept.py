"""What Fitting keeps of its results: every solution of Predict parameters saved with its probability, flags and
intervals (Save ▸ Solutions…, Save ▸ Data and Fit's record) and its ± in the values table; a refinement of a solution
said as such; projects that keep the last fit, the solutions and a series' fitted frames; Predict's stages and what
Stop keeps; Undo and Redo that say what they did. Predict is stood in for (``tests/test_fit_page.PredictJobs``)."""

from __future__ import annotations

import csv
import json
import threading
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import FitData, model_to_dict, residuals
from tests.test_fit_page import _curve_file, _page, _predicting, _sphere, _wait

INTERVALS = {"R_1": [None, None, None, None, None, None, None], "D_1": [40.0, 40.5, 38.0, 43.0, 36.0, 45.0, 1.5],
             "sigma_R_1": [0.15, 0.15, 0.12, 0.18, 0.10, 0.21, 0.02]}


def _row(payload, radius, probability, flags=()):
    data = FitData.prepare(payload["q"], payload["intensity"], payload["sigma"])
    model = _sphere(radius)
    r = residuals(model, data)
    values = dict(INTERVALS, R_1=[radius, radius + 0.05, radius - 0.2, radius + 0.3, radius - 0.5, radius + 0.6, 0.1])
    return {"composition": 0, "name": "sphere (S)", "probability": probability, "model": model_to_dict(model),
            "chi2": float(r @ r) / (data.q.size - 6), "values": values, "flags": list(flags), "ess": 120.0,
            "points": {"fitted": int(data.q.size), "given": int(data.q.size), "below_window": 0, "unusable": 0}}


def _answer(payload, *_rest):
    return {"solutions": [_row(payload, 6.0, 0.6, ["at_bound:tail_exponent"]), _row(payload, 5.5, 0.4, ["low_ess"])],
            "tried": None, "network": {"name": "flow_test", "sha256": "0" * 64}, "seconds": 1.5}


def _predicted(tmp_path):
    page = _page()
    _predicting(page, _answer)
    assert page.open_curve(_curve_file(tmp_path))
    page.set_method("predict")
    page.run_fit()
    _wait(page)
    assert len(page.session.solutions) == 2, page.status_label.text()
    return page


def _cells(page, name):
    table = page.parameters_table
    line = next(line for line in range(table.rowCount()) if table.item(line, 0).text() == name)
    return table.item(line, 1).text(), table.item(line, 2).text(), table.item(line, 2).toolTip()


def test_predict_solutions_have_their_interval_as_plus_minus_and_distinct_names(tmp_path) -> None:
    page = _predicted(tmp_path)
    try:
        value, plus, tip = _cells(page, "R")
        assert plus == "± 0.25" and "half the width of the 68 % posterior interval [5.8, 6.3]" in tip
        assert _cells(page, "Scale")[1] == "–" and "solves the scales" in _cells(page, "Scale")[2]
        texts = [page.solution_selector.itemText(index) for index in range(page.solution_selector.count())]
        assert texts[0].startswith("1: Sphere #1 · χ²ᵣ") and "p 0.60" in texts[0] and "at bound: Tail ν" in texts[0]
        assert texts[1].startswith("2: Sphere #2") and "few samples" in texts[1]
        assert "stopped at a bound" in page.solution_selector.itemData(0, 3)  # the flags' sentences on hover
        assert [page.solutions_table.item(line, 1).text() for line in range(2)] == ["Sphere #1", "Sphere #2"]
    finally:
        page.dispose()


def test_solutions_are_saved_with_every_value_and_interval(tmp_path) -> None:
    page = _predicted(tmp_path)
    try:
        assert page.export_solutions_action.isEnabled() and page.export_solutions_button.isVisibleTo(page)
        saved = page.export_solutions_dialog(str(tmp_path / "sample_solutions.csv"))
        assert saved, page.status_label.text()
        with open(tmp_path / "sample_solutions.csv", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        r = [row for row in rows if row["parameter"] == "0:R"]
        assert [row["solution"] for row in r] == ["1", "2"] and r[0]["label"] == "sphere (S) #1"
        assert float(r[0]["probability"]) == 0.6 and r[0]["flags"] == "at_bound:tail_exponent"
        assert float(r[0]["low68"]) == pytest.approx(5.8) and float(r[0]["high95"]) == pytest.approx(6.6)
        assert r[0]["in_model"] == "1" and r[1]["in_model"] == "0"
        background = next(row for row in rows if row["parameter"] == "globals:background")
        assert background["value"] and background["low68"] == ""  # no interval: solved for every sample
        record = json.loads((tmp_path / "sample_solutions.json").read_text(encoding="utf-8"))
        block = record["predict"]
        assert [s["probability"] for s in block["solutions"]] == [0.6, 0.4] and block["in_model"] == 1
        assert block["run"]["network"]["name"] == "flow_test" and block["solutions"][1]["flags"] == ["low_ess"]
        assert block["solutions"][0]["points"]["fitted"] == 240 and "half the width" in block["plus_minus"]
        with open(tmp_path / "sample_solutions_curves.csv", encoding="utf-8") as stream:
            header = next(csv.reader(stream))
        assert header[:4] == ["q_nm^-1", "q_A^-1", "I", "sigma"] and "s1_model" in header and "s2_residual" in header
        assert any(name.startswith("s2_1") for name in header)  # the terms of every solution
        assert page.export_data_dialog(str(tmp_path / "fit.csv"))
        assert len(json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))["predict"]["solutions"]) == 2
    finally:
        page.dispose()


def test_a_refinement_of_a_solution_says_so_and_undo_says_what_it_took_back(tmp_path) -> None:
    page = _predicted(tmp_path)
    try:
        page.run_fit("local")  # Refine These Values
        _wait(page)
        assert page.session.refined_from == {"solution": 1, "method": "local", "edited": False}
        assert "Values from a new refinement (Refine) of solution 1, Sphere #1: not the values Predict gave" \
            in page.quality_label.text()
        assert page.solution_selector.itemText(0).startswith("Refined from solution 1 (Refine) · χ²ᵣ")
        assert page.solution_selector.currentIndex() == 0 and page.solution_selector.itemData(0) is None
        assert page.undo_button.toolTip() == "Undo: the fit (Refine) (Ctrl+Z)"
        page.undo()
        assert page.status_label.text() == "Undone: the fit (Refine)."
        assert page.redo_button.toolTip() == "Redo: the fit (Refine) (Ctrl+Shift+Z)"
        assert not page.solution_selector.itemText(0).startswith("Refined")  # the solution is in Model again
        page.redo()
        assert page.status_label.text() == "Redone: the fit (Refine)."
        page.solutions_table.selectRow(1)
        page.use_selected_solution()
        page.undo()
        assert page.status_label.text() == "Undone: solution 2 in Model."
    finally:
        page.dispose()


def test_a_project_keeps_the_fit_and_the_solutions(tmp_path) -> None:
    page = _predicted(tmp_path)
    other = _page()
    try:
        page.run_fit("local")
        _wait(page)
        state = json.loads(json.dumps(page.project_state()))  # as File ▸ Save Project writes it
        notes = other.apply_project_state(state)
        assert notes == []
        session, saved = other.session, page.session
        assert session.result is not None and session.result_is_current()
        assert session.result.chi2_reduced == saved.result.chi2_reduced and session.result.errors == saved.result.errors
        assert session.result.at_bounds == saved.result.at_bounds and session.result.method == "local"
        assert [s.probability for s in session.solutions] == [0.6, 0.4] and session.solutions[0].flags == saved.solutions[0].flags
        assert session.solutions[0].intervals == saved.solutions[0].intervals and session.solutions_are_current()
        assert session.refined_from == saved.refined_from and session.predict_run["network"]["name"] == "flow_test"
        assert other.quality_label.text() == page.quality_label.text() and other.step_rail.state("fit") == "ok"
        assert other.parameters_table.item(0, 2).text().startswith("± ")
        curve = _curve_file(tmp_path)  # the file changed since: the results are shown, stale
        np.savetxt(curve, np.loadtxt(curve) * [1.0, 1.01, 1.0])
        third = _page()
        try:
            notes = third.apply_project_state(state)
            assert any("changed since the project was saved" in note for note in notes)
            assert third.session.result is not None and not third.session.result_is_current()
            assert third.step_rail.detail("fit") == "Changed after the fit: Fit again"
        finally:
            third.dispose()
    finally:
        page.dispose()
        other.dispose()


class _StagedJobs:
    """A Predict job that reports its stages and the solutions finished, then waits until it is cancelled."""

    def __init__(self, partial):
        self.partial, self.stop = partial, threading.Event()

    def run(self, request, on_progress):
        from src.gimap.app.jobs import JobError, JobProgress, JobResult

        for stage, done, total, extra in (("loading", 0, 1, {}), ("proposals", 2, 4, {}), ("refinement", 4, 4, {}),
                                          ("sampling", 1, 3, {"partial": self.partial(request.payload)})):
            on_progress(JobProgress(request.job_id, 0.6, 1, stage, {"stage": stage, "done": done, "total": total,
                                                                    "elapsed": 65.0, "left": 30.0, **extra}))
        self.stop.wait(30)
        return JobResult(request.job_id, "cancelled", error=JobError("cancelled", "Job was cancelled."))

    def cancel(self, _job_id):
        self.stop.set()
        return True


def _running(page, jobs):
    page.predict_available = lambda: True
    page.view_model.context.jobs = jobs
    page.set_method("predict")
    page.run_fit()
    end = time.monotonic() + 10
    while time.monotonic() < end and "Sampling" not in page.fit_progress_text.text():
        QApplication.processEvents()
        time.sleep(0.01)


def test_predict_shows_its_stages_and_stop_keeps_the_solutions_finished(tmp_path) -> None:
    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        jobs = _StagedJobs(lambda payload: [_row(payload, 6.0, 1.0)])
        _running(page, jobs)
        line = page.fit_progress_text.text()
        assert line.startswith("Step 3 of 3 · Sampling the posterior: solution 2 of 3 · 1:05 elapsed"), line
        assert "about 0:30 left in this step" in line and page.fit_progress.maximum() == 100
        page.stop_fit()
        assert page.fit_progress_text.text() == "Stopping … the solutions finished so far are kept."
        _wait(page)
        solutions = page.session.solutions
        assert len(solutions) == 1 and solutions[0].incomplete and page.session.predict_run["finished"] == 1
        assert page.status_label.text().startswith("Stopped: 1 solution kept, of the 1 of 3 solutions sampled")
        assert "incomplete" in page.solution_selector.itemText(0) and "Incomplete" in page.quality_label.text()
        assert page.export_solutions_dialog(str(tmp_path / "partial.csv"))
        record = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        assert record["predict"]["solutions"][0]["incomplete"] is True
        assert record["predict"]["probabilities"].startswith("over the solutions finished")
        empty = _StagedJobs(lambda payload: [])  # nothing finished: nothing changes
        _running(page, empty)
        page.stop_fit()
        _wait(page)
        assert page.session.solutions == solutions
        assert page.status_label.text() == "Stopped before any solution was finished: no new solutions."
    finally:
        page.dispose()


def test_the_job_reports_stages_and_partial_solutions_without_changing_the_run(monkeypatch) -> None:
    from src.gimap.features.fitting.infrastructure.adapters import posterior_job

    reports, calls = [], {}

    def predict(network, q, intensity, sigma, **kwargs):
        calls.update(kwargs)
        kwargs["progress"]("proposals", 0, 2)
        kwargs["progress"]("sampling", 0, 2)
        kwargs["partial"]([])
        kwargs["progress"]("sampling", 2, 2)
        return []

    monkeypatch.setattr(posterior_job.posterior_network, "load", lambda: type("N", (), {"meta": {}})())
    monkeypatch.setattr(posterior_job.posterior_network, "path", lambda: __import__("pathlib").Path("flow.npz"))
    monkeypatch.setattr(posterior_job.refine, "predict", predict)
    q = np.linspace(0.1, 2.0, 50)
    value = posterior_job.run_predict({"q": q.tolist(), "intensity": (1 / q).tolist(), "sigma": None, "qz": 0.0},
                                      lambda *report: reports.append(report), lambda: False)
    assert value["solutions"] == [] and callable(calls["progress"]) and callable(calls["partial"])
    stages = [(r[3]["stage"], r[3]["done"]) for r in reports if len(r) > 3]
    assert stages == [("loading", 0), ("proposals", 0), ("sampling", 0), ("sampling", 1), ("sampling", 2)]
    assert reports[3][3]["partial"] == [] and reports[3][2] == "Sampling the posterior: solution 2 of 2"
    assert [round(r[0], 3) for r in reports if len(r) > 3] == [0.0, 0.02, 0.55, 0.775, 1.0]


def test_a_project_keeps_the_fitted_frames_of_a_series(tmp_path) -> None:
    from tests.test_fit_series import _loaded, _model, _open_series, _pages, _series
    from tests.test_fit_series import _wait as _wait_series

    single, series = _pages()
    folder = _series(tmp_path, radii=(5.0, 5.4, 5.8))
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert _open_series(series, folder)
    series.start()
    _wait_series(series)
    assert len(series.fits) == 3 and all(frame.ok for frame in series.fits.values())
    state = json.loads(json.dumps(series.project_state()))
    other_single, other = _pages()
    try:
        assert other.apply_project_state(state) == []
        _loaded(other)
        assert sorted(other.fits) == [0, 1, 2] and other.results_table.rowCount() == 3
        for index, frame in series.fits.items():
            again = other.fits[index]
            assert again.result.chi2_reduced == frame.result.chi2_reduced and again.result.errors == frame.result.errors
            assert again.points == frame.points and again.halves_used == frame.halves_used
        assert other._start_model == series._start_model and "3 fitted frames" in other.status_label.text()
    finally:
        for page in (single, series, other_single, other):
            page.dispose()
