"""The actual single-curve page uses the v2 engine and keeps its scientific result intact."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from src.gimap.app.jobs import JobError, JobResult
from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component
from src.gimap.features.fitting.application.workflow_v5 import fit1d_kit_home
from tests.test_fit_page import _page, _wait

needs_kit = pytest.mark.skipif(
    not (fit1d_kit_home() / "gisaxs1d_kit" / "core.py").is_file(),
    reason="the gisaxs1d kit (GISAXS1D_KIT_HOME or utils/gisaxs1d_gui_kit) is not installed",
)


def _synthetic_file(tmp_path, *, sigma=True):
    q = np.linspace(0.06, 1.8, 48)
    y = 5.0 + 300.0 * np.exp(-1.4 * q)
    errors = np.sqrt(y)
    path = tmp_path / "synthetic.dat"
    np.savetxt(path, np.column_stack([q / 10, y] + ([errors] if sigma else [])))
    return path


@needs_kit
@pytest.mark.parametrize("network,mode,count", [("general", "gisaxs", 1), ("au_np_si", "gisaxs", 2),
                                              ("general", "saxs", 3), ("au_np_si", "saxs", 4)])
def test_main_fit_runs_without_tf_and_preserves_native_result(tmp_path, monkeypatch, network, mode, count):
    page = _page()
    try:
        assert page.open_curve(_synthetic_file(tmp_path, sigma=count != 3)), page.status_label.text()
        families = ["sphere", "vertical_cylinder", "sphere", "vertical_cylinder"][:count]
        page.set_model(FitModel(tuple(new_component(family) for family in families)))
        page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("model"))
        page.fit1d_network.setCurrentIndex(page.fit1d_network.findData(network))
        page.fit1d_mode.setCurrentIndex(page.fit1d_mode.findData(mode))
        page.fit1d_policy.setCurrentIndex(page.fit1d_policy.findData("fast"))
        page.fit1d_use_qz.setChecked(count != 2)
        page.fit1d_qz.setValue(0.42)
        page.fit1d_sigma_floor.setValue(3.0)
        assert "fit1d uses σ from the file" in page.sigma_note.text() or count == 3
        if count == 3:
            assert "Poisson errors" in page.sigma_note.text()
        # This path must not even ask for the old V5 model directory.
        monkeypatch.setattr("src.gimap.features.fitting.presentation.single.run.bundled_workflow",
                            lambda: pytest.fail("fit1d must not require V5/TensorFlow assets"))
        page.run_fit("fit1d")
        _wait(page)
        native = page.session.native_result
        assert native is not None, page.status_label.text()
        assert page.session.native_is_current() and page.session.result is None
        assert native["summary"]["model"] == network
        assert native["summary"]["mode"] == mode
        assert native["summary"]["qz_nm_inv"] == pytest.approx(0.42 if mode == "gisaxs" and count != 2 else 0)
        if count == 2:
            assert any("No qz" in warning for warning in native["warnings"])
        if count == 3:
            assert any("Poisson errors" in warning for warning in native["warnings"])
            assert not any("Relative σ" in warning for warning in native["warnings"])
        assert len(native["populations"]) == count
        assert native["conventions"]["physics"].startswith("v2")
        curves = native["curves"]
        q, y = np.asarray(curves["q_nm_inv"]), np.asarray(curves["intensity"])
        assert q == pytest.approx(np.linspace(0.06, 1.8, 48))
        assert curves["sigma"] == pytest.approx(np.hypot(np.sqrt(y), 0.03 * y))
        terms = sum(np.asarray(term) for term in curves["components"].values())
        assert terms == pytest.approx(curves["model"])
        # The mapped manual model is an initial guess; its forward curve must never replace v2.
        assert not np.allclose(evaluate(page.session.model, q), curves["model"], rtol=1e-4)
        page.terms_check.setChecked(True)
        shown = page.plot._source  # the scientific arrays supplied to the plot, before logarithmic display
        assert len(shown) >= count + 2
        assert any(np.allclose(np.asarray(item[2]), curves["model"]) for item in shown if len(item[2]) == len(q))
        assert page.parameters_table.rowCount() == len(native["table"])
        assert "χ²/point" in page.quality_label.text()
        assert "starting point" in page.quality_label.text()
        output = tmp_path / "native_fit.csv"
        assert page.export_data_dialog(output) == output
        exported = np.loadtxt(output, delimiter=",", skiprows=1)
        assert exported[:, 3] == pytest.approx(curves["sigma"], rel=1e-7)
        assert exported[:, 4] == pytest.approx(curves["model"], rel=1e-7)
        assert exported[:, 5] == pytest.approx(curves["residual"], abs=1e-7)
        record = json.loads(output.with_suffix(".json").read_text())
        assert record["backend"] == "fit1d_physics_v2"
        assert record["native_result"]["warnings"] == native["warnings"]
        mapped = page.session.model
        page.set_model(mapped.with_values({(0, "R"): mapped.components[0].value("R") * 1.02}))
        assert not page.session.native_is_current()
        assert "manual starting model" in page.quality_label.text()
        assert "previous fit1d result" in page.quality_label.text()
        assert "original fit1d v2 curve is shown" not in page.quality_label.text()
        page.undo()
        assert page.session.native_is_current()
        page.fit1d_sigma_floor.setValue(4.0)
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "changed_floor.csv") is None
        page.fit1d_sigma_floor.setValue(3.0)
        assert page.session.native_is_current()
        for widget in (page.fit1d_mode, page.fit1d_network, page.fit1d_policy):
            original = widget.currentIndex()
            widget.setCurrentIndex((original + 1) % widget.count())
            assert not page.session.native_is_current()
            assert page.export_data_dialog(tmp_path / "changed_settings.csv") is None
            widget.setCurrentIndex(original)
            assert page.session.native_is_current()
        checked = page.fit1d_use_qz.isChecked()
        page.fit1d_use_qz.setChecked(not checked)
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "changed_qz.csv") is None
        page.fit1d_use_qz.setChecked(checked)
        assert page.session.native_is_current()
        page.set_range((0.2, 1.0))
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "stale.csv") is None
    finally:
        page.dispose()


def test_fit1d_rejects_unknown_families_and_five_populations(tmp_path):
    page = _page()
    try:
        page.open_curve(_synthetic_file(tmp_path))
        for families in (["cylinder"], ["sphere"] * 5):
            page.set_model(FitModel(tuple(new_component(f) for f in families)))
            page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("model"))
            page.run_fit("fit1d")
            assert not page.fit_running() and page.session.native_result is None
            assert "1–4 particles" in page.status_label.text()
    finally:
        page.dispose()


def test_stop_reaches_jobrunner_without_a_new_result(tmp_path):
    class BlockingRunner:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.request = None

        def run(self, request, on_progress):
            self.request = request
            self.started.set()
            assert self.stopped.wait(5), "Stop did not reach JobRunner"
            return JobResult(request.job_id, "cancelled", error=JobError("cancelled", "Cancelled"))

        def cancel(self, job_id):
            assert self.request.job_id == job_id
            self.stopped.set()
            return True

    page = _page()
    runner = BlockingRunner()
    page.view_model.context.jobs = runner
    try:
        page.open_curve(_synthetic_file(tmp_path))
        page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("model"))
        page.run_fit("fit1d")
        end = time.monotonic() + 5
        while not runner.started.is_set() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        assert runner.started.is_set()
        page.stop_fit()
        _wait(page)
        assert runner.stopped.is_set() and page.session.native_result is None
        assert "Stopped" in page.status_label.text()
    finally:
        page.dispose()


# -- Predict parameters: its own action, candidates, constraints and edges ------------------------------------------


class _Recorder:
    """A job runner that keeps the request and ends at once (no fit)."""

    def __init__(self):
        self.requests = []

    def run(self, request, on_progress):
        self.requests.append(request)
        return JobResult(request.job_id, "cancelled", error=JobError("cancelled", "Cancelled"))

    def cancel(self, job_id):
        return True


def _signed_file(tmp_path, name="cut.dat"):
    from tests.test_fit1d_method import _example

    q, y, s = _example()
    path = tmp_path / name
    both = np.column_stack([np.r_[-q[::-1], q] / 10, np.r_[y[::-1], y], np.r_[s[::-1], s]])
    np.savetxt(path, both)
    return path


def _unsigned_file(tmp_path, name="half.dat"):
    from tests.test_fit1d_method import _example

    q, y, s = _example()
    path = tmp_path / name
    np.savetxt(path, np.column_stack([q / 10, y, s]))
    return path


def test_one_method_choice_recommends_predict_for_an_unfitted_gisaxs_cut(tmp_path):
    page = _page()
    runner = _Recorder()
    page.view_model.context.jobs = runner
    try:
        assert page.open_curve(_signed_file(tmp_path))
        assert page.recommended_method() == "fit1d" and page.method() == "fit1d"
        assert page.step_stack.currentWidget() is page.step_pages["fit"]
        assert page.fit_step_button.text() == "Predict parameters (fit1d)"
        assert page.method_hint.text().startswith("Recommended for this curve")
        assert page.fit1d_row.isVisibleTo(page) and not page.families_row.isVisibleTo(page)
        assert page.fit1d_composition.currentData() == "auto:sphere,vc:3"  # no particles of the user's yet
        page.fit_button.click()  # the command bar's Fit runs the method chosen
        _wait(page)
        payload = runner.requests[-1].payload
        assert payload["composition"] == "auto:sphere,vc:3" and payload["constraints"] == [] and payload["rounds"] == 2
        page.resolve_edges_check.setChecked(False)
        page.run_fit("fit1d")
        _wait(page)
        assert runner.requests[-1].payload["rounds"] == 0
        page.resolve_edges_check.setChecked(True)
        # Model's constraints go to fit1d with the particles in Model; a narrow range is said before the fit.
        from src.gimap.features.fitting.application.single_fit import FitModel, new_component

        model = FitModel((new_component("vertical_cylinder"), new_component("sphere")))
        model = model.with_parameter((1, "R"), lower=2.0, upper=900.0).with_parameter((0, "sigma_R"), value=0.1,
                                                                                      free=False)
        page.set_model(model)
        page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("model"))
        assert page.fit1d_notes.isVisibleTo(page) and "beyond what this curve can resolve" in page.fit1d_notes.text()
        page.run_fit()
        _wait(page)
        constraints = {item["name"]: item for item in runner.requests[-1].payload["constraints"]}
        assert constraints["pop1.R_nm"]["kind"] == "range" and constraints["pop1.R_nm"]["high"] == 900.0
        assert constraints["pop2.sigma_R_rel"]["kind"] == "fixed" and constraints["pop2.sigma_R_rel"]["low"] == 0.1
        page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("auto:vc:3"))
        assert "not passed" in page.fit1d_notes.text()
        page.set_method("local")
        assert not page.fit1d_row.isVisibleTo(page) and not page.fit1d_notes.isVisibleTo(page)
        assert not page.method_hint.text().startswith("Recommended")
    finally:
        page.dispose()
    page = _page()
    try:
        assert page.open_curve(_unsigned_file(tmp_path))  # a curve that is not known to be a GISAXS cut
        assert page.recommended_method() == "" and page.method() != "fit1d"
    finally:
        page.dispose()


@needs_kit
def test_compositions_are_compared_and_one_is_predicted_again_then_refined(tmp_path):
    page = _page()
    try:
        assert page.open_curve(_signed_file(tmp_path))
        page.fit1d_policy.setCurrentIndex(page.fit1d_policy.findData("fast"))
        page.run_fit()
        _wait(page)
        native = page.session.native_result
        assert native is not None, page.status_label.text()
        assert page.method() == "local" and page.next_refine_button.isVisibleTo(page)  # next: refine
        assert page.alternatives.isVisibleTo(page) and "Other compositions" in page.alternatives.toggle_button.text()
        table = page.fit1d_candidates_table
        assert table.rowCount() >= 2
        assert table.item(0, 1).text() == native["summary"]["composition"] and "in Results" in table.item(0, 4).text()
        alike = [line for line in range(table.rowCount()) if "indistinguishable from" in table.item(line, 4).text()]
        assert alike == [i for i, c in enumerate(page._fit1d_candidates) if c["indistinguishable_from"]]
        assert "Compositions tried" in page.fit1d_constraints_label.text()
        table.selectRow(1)
        second = page._fit1d_candidates[1]
        details = page.fit1d_candidate_details
        assert details.rowCount() == len(second["table"]) and "±" in "".join(
            details.item(line, 1).text() for line in range(details.rowCount()))
        page.fit1d_use_candidate_button.click()  # predicts again with this composition
        _wait(page)
        native = page.session.native_result
        assert native["summary"]["composition"] == second["composition"] and page.session.native_is_current()
        names = {"sphere": "sphere", "vc": "vertical_cylinder"}
        assert [c.family for c in page.session.model.components] == [names[s] for s in second["composition"].split(",")]
        assert len(page._fit1d_candidates) >= 2 and "in Results" in table.item(1, 4).text()  # the list stays
        output = tmp_path / "candidate.csv"
        assert page.export_data_dialog(output) == output
        record = json.loads(output.with_suffix(".json").read_text())
        assert record["native_result"]["fit1d_constraints"]["composition"] == "model"
        page.next_refine_button.click()
        _wait(page)
        assert page.session.result is not None and page.session.result.method == "local", page.status_label.text()
        assert page.fit_again_button.isVisibleTo(page) and not page.next_refine_button.isVisibleTo(page)
    finally:
        page.dispose()


@needs_kit
def test_values_at_an_edge_are_resolved_by_the_fit_itself(tmp_path):
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    page = _page()
    try:
        assert page.open_curve(_unsigned_file(tmp_path))
        page.fit1d_policy.setCurrentIndex(page.fit1d_policy.findData("fast"))
        page.fit1d_use_qz.setChecked(True)
        page.fit1d_qz.setValue(0.5)
        model = FitModel((new_component("vertical_cylinder"), new_component("sphere")))
        page.set_model(model.with_parameter((1, "R"), value=1.5, lower=1.0, upper=2.0))
        page.fit1d_composition.setCurrentIndex(page.fit1d_composition.findData("model"))
        page.run_fit("fit1d")
        _wait(page)
        native = page.session.native_result
        assert native is not None, page.status_label.text()
        sphere = next(p for p in native["populations"] if p["name"] == "pop1")
        assert sphere["R_nm"] == pytest.approx(3.0, rel=0.1)  # the true radius, beyond the range given
        used = {item["name"]: item for item in native["fit1d_constraints"]["used"]}
        assert used["pop1.R_nm"]["source"] == "widened" and used["pop1.R_nm"]["high"] >= 4.0
        assert "pop1.R_nm" not in {edge["name"] for edge in native["fit1d_bounds"]["edges"]}
        assert page.actions_label.isVisibleTo(page)
        text = page.actions_label.text()
        assert text.startswith("Done automatically") and "edge of your range (2 nm): widened to 4 nm" in text
        assert "(Model)" in page.fit1d_constraints_label.text() or "(widened here)" in page.fit1d_constraints_label.text()
        for action in native["fit1d_bounds"]["actions"]:
            if "fits the curve as well" in action["text"]:  # a value that stayed at an edge: given as a bound
                assert any(item["source"].startswith("bound_") and item["kind"] == "fixed" for item in used.values())
                assert any(row["status"] == "one_sided" for row in native["table"])
        assert not page.findChildren(QPushButton, "fit1dEdge_widen")  # nothing to click
    finally:
        page.dispose()


def _sphere_curve(tmp_path, radius=6.0):
    from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component

    q = np.linspace(0.08, 2.0, 160)
    truth = FitModel((new_component("sphere", radius=radius),)).with_parameter((0, "Int"), value=5e4)
    clean = evaluate(truth, q) + 2.0
    noisy = clean + np.random.default_rng(7).normal(0, 1, q.size) * np.sqrt(clean)
    path = tmp_path / "sphere_fit_input.dat"
    np.savetxt(path, np.column_stack([q / 10, noisy, np.sqrt(clean)]))
    return path


def test_refine_resolves_a_range_edge_and_the_values_are_fixed_from_the_results(tmp_path):
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    page = _page()
    asked = []
    page.seriesRequested.connect(lambda folder, pattern: asked.append((folder, pattern)))
    try:
        assert page.open_curve(_sphere_curve(tmp_path))
        start = FitModel((new_component("sphere", radius=4.0),)).with_parameter((0, "R"), lower=2.0, upper=4.5)
        page.set_model(start)
        page.set_method("local")
        page.run_fit()
        _wait(page)
        result = page.session.result
        assert result is not None and page.session.model.get((0, "R")).value == pytest.approx(6.0, rel=0.05)
        assert page.session.model.get((0, "R")).upper == 9.0  # widened by the fit itself
        assert page.actions_label.isVisibleTo(page) and "widened to 9" in page.actions_label.text()
        assert "stopped at a bound" not in page.warnings_label.text()
        # From the results: fix R, fit again with it fixed; the series entry lists this folder.
        line = page._row_paths.index((0, "R"))
        page.parameters_table.clearSelection()
        page.parameters_table.selectRow(line)
        assert page.fix_button.isEnabled() and not page.free_button.isEnabled()
        page.fix_button.click()
        assert not page.session.model.get((0, "R")).free and "fixed in Model" in page.status_label.text()
        page.fit_again_button.click()
        _wait(page)
        assert page.session.result.method == "local" and not page.session.result.model.get((0, "R")).free
        page.series_button.click()
        assert asked == [(str(tmp_path), "*_fit_input.dat")]
        page.parameters_table.clearSelection()  # a Ctrl left held by another test would make selectRow toggle
        page.parameters_table.selectRow(page._row_paths.index((0, "R")))
        page.range_button.click()
        assert page.step_stack.currentWidget() is page.step_pages["model"] and page.show_ranges_check.isChecked()
    finally:
        page.dispose()


@needs_kit
def test_the_series_predicts_every_frame_and_follows_the_previous_one(tmp_path):
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component
    from tests.test_fit1d_method import _cylinders
    from tests.test_fit_series import _pages, _wait as _wait_series

    folder = tmp_path / "series"
    folder.mkdir()
    radii = (3.0, 3.1, 3.2, 3.3)
    for frame, radius in enumerate(radii):
        q, y, s = _cylinders(radius=radius, distance=30.0, seed=frame)
        np.savetxt(folder / f"run_{frame + 1:05d}_fit_input.dat", np.column_stack([q / 10, y, s]))
    single, series = _pages()
    try:
        assert single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(FitModel((new_component("vertical_cylinder"),)))
        single.fit1d_policy.setCurrentIndex(single.fit1d_policy.findData("fast"))
        series.open_series(folder)
        series.set_series_method("fit1d")
        assert series.series_method() == "fit1d" and series.follow_check.isEnabled() and series.follow_check.isChecked()
        series.start()
        _wait_series(series, 600)
        frames = [series.fits[index] for index in sorted(series.fits)]
        assert len(frames) == 4 and all(frame.ok for frame in frames), [frame.error for frame in frames]
        assert all(frame.native is not None and frame.result.method == "fit1d" for frame in frames)
        fitted = [frame.native["populations"][0]["R_nm"] for frame in frames]
        assert fitted == pytest.approx(radii, rel=0.1)
        assert "fit1d_follow" not in frames[0].native  # nothing to follow at the first frame
        for frame in frames[1:]:
            follow = frame.native["fit1d_follow"]
            assert follow["followed"] and follow["chi2_follow"] - follow["chi2_free"] <= follow["tolerance"]
            assert any("previous frame" in text for text in frame.resolved)
        # The table has the frames' values with the kit's errors, in Fitting's names.
        assert frames[1].result.model.get((0, "R")).value == pytest.approx(fitted[1])
        assert frames[1].result.errors.get((0, "R")) == pytest.approx(frames[1].native["populations"][0]["R_nm_err"])
        path = series.save_table(str(tmp_path / "series.csv"))
        record = json.loads(open(path.replace(".csv", ".json"), encoding="utf-8").read())
        assert record["settings"]["method"] == "fit1d" and record["settings"]["follow"] is True
        assert [frame.get("followed_previous") for frame in record["per_frame"]] == [None, True, True, True]
    finally:
        series.dispose()
        single.dispose()
