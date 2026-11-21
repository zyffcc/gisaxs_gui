"""GISAXS in the automatic analysis: Yoneda cut, symmetry, halves, length scale 2π/q*, fit, report and results panel.

The real frame is BornAgain's GALAXI example (tests/data/external): silver spheres in a film, a
radial paracrystal with a peak distance of 53.6 nm in BornAgain's model of it. The shoulder of its cut
lies near 2π/53.6 nm only by coincidence: BornAgain simulations of the same model without interference
show a shoulder too, so 2π/q* is reported as a length scale of the cut, never as the particle spacing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QLabel, QPushButton

from src.gimap.features.assistant.application import PipelineOptions, compact_report, pipeline_markdown
from src.gimap.features.assistant.domain import choose_halves, spacing_from_peaks
from tests.test_assistant_gui import _app

DATA = Path(__file__).parent / "data" / "external"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
GALAXI = MANIFEST["gisaxs_galaxi"]
SHOULDER_Q = 0.0113
"""|qy| (Å⁻¹) of the shoulder of the GALAXI cut (a regression value of the public frame, not a spacing)."""


def _geometry() -> dict:
    return {
        "distance_mm": GALAXI["distance_m"] * 1e3, "beam_center_px": GALAXI["beam_center_px"],
        "wavelength_angstrom": GALAXI["wavelength_angstrom"], "energy_kev": 12.398419843 / GALAXI["wavelength_angstrom"],
        "pixel_size_um": [GALAXI["pixel_size_m"] * 1e6] * 2, "source_image": "galaxi_geometry.json",
        "shape": [1043, 981], "from_file": True,
    }


class FakeFitter:
    """Two solutions of Predict parameters, both probable (p 0.62 and 0.38); records what it was asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, intensity, sigma, *, components=(), report=None, cancelled=None, cut_qz=None):
        self.calls.append({"points": len(q), "components": components, "q_min": float(np.min(q)), "cut_qz": cut_qz})
        return [_row(q, intensity, sigma, 1, "random_cylinder", 6.1, 398.0, 1.22, probability=0.62),
                _row(q, intensity, sigma, 2, "sphere", 9.4, 53.4, 1.31, probability=0.38)]


UNCONSTRAINED = "the curve says little about σh/h (its 68 % interval spans most of its allowed range)"
"""A note of Predict parameters as the report says it: the flag ``unconstrained:sigma_h_1`` of Fitting's row in the
words and with the name of ``shared/fit_names`` (one population: no number)."""


class BoundFitter(FakeFitter):
    """Solutions of Predict parameters with values at a bound of their range (invented values of the kind GALAXI
    gives): the most probable has D at its upper bound, the second (p 0.25) σh/h at its lower bound, the third (a
    sphere, D = 48.3484 nm) none."""

    def __call__(self, q, intensity, sigma, *, components=(), report=None, cancelled=None, cut_qz=None):
        self.calls.append({"points": len(q), "components": components, "q_min": float(np.min(q))})
        upper_d = [{"parameter": "D", "bound": "upper", "component": 0},
                   {"parameter": "sigma_h", "bound": "upper", "component": 0}]
        return [
            _row(q, intensity, sigma, 1, "random_cylinder", 8.5961, 500.0, 1.26, upper_d, probability=0.55,
                 flags=["at_bound:D_1", "unconstrained:sigma_h_1"]),
            _row(q, intensity, sigma, 2, "random_cylinder", 6.0117, 30.862, 1.279,
                 [{"parameter": "sigma_h", "bound": "lower", "component": 0}], probability=0.25),
            _row(q, intensity, sigma, 3, "sphere", 9.4416, 48.3484, 1.358, probability=0.15),
            _row(q, intensity, sigma, 4, "vertical_cylinder", 7.7732, 30.8046, 1.45,
                 [{"parameter": "nu_Res", "bound": "upper", "component": None}], probability=0.05),
        ]


def _row(q, intensity, sigma, rank, model, radius, distance, chi2, at_bounds=(), *, probability=1.0, warnings=None,
         flags=None):
    """One solution as Fitting's Predict parameters gives it (q in nm⁻¹; ``at_bounds``: the values that ended on a
    bound of their range; ``posterior``: best, median, 68 % and 95 % limits, error of each value)."""
    grid = np.linspace(0.05, 2.5, 50)
    native = 10.0 * np.asarray(q, dtype=float)
    return {
        "rank": rank, "combination": model, "composition": model, "probability": probability,
        "best_chi2_weighted": chi2, "best_log_rmse": 0.15, "converged": True,
        "warnings": list(warnings or ()), "at_bounds": list(at_bounds),
        **({} if flags is None else {"flags": list(flags)}),  # as Fitting's rows carry them (a value at a bound too)
        "posterior": {"R_1": [radius, radius, 0.97 * radius, 1.03 * radius, 0.94 * radius, 1.06 * radius, 0.02],
                      "D_1": [distance, distance, 0.9 * distance, 1.1 * distance, 0.8 * distance, 1.2 * distance, 1.0]},
        "display_q": grid.tolist(), "display_fit": (1e3 / (1 + (grid * radius) ** 4)).tolist(),
        "components": [{"type": model, "weight": 1.0, "amplitude": 2.5e4,
                        "params": {"R": radius, "sigma_R": 0.3, "D": distance, "sigma_D": 0.4}}],
        "global_params": {"background": 0.8, "resolution_amplitude": 0.0, "sigma_Res": 0.01, "nu_Res": 1.0},
        "nfev": None, "seconds": 13.2, "conditions_source": "network_choice",
        "algorithm": "Predict parameters: the proposal network's importance-weighted samples, each mode refined by "
                     "bounded least squares, the posterior sampled around the best solutions",
        "native_q": native.tolist(), "native_fit": (1e3 / (1 + (native * radius) ** 4)).tolist(),
        "observed": list(map(float, intensity)), "sigma": list(map(float, sigma)),
    }


@pytest.fixture(scope="module")
def galaxi_run():
    from src.gimap.app.headless_assistant import open_headless_session

    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    session.catalog.fitter = FakeFitter()
    report = session.run_pipeline(PipelineOptions(technique="gisaxs", geometry=_geometry(), incidence_deg=GALAXI["incidence_deg"]))
    yield session, report
    session.close()


def _decision(report: dict, what: str) -> dict:
    return next(item for item in report["decisions"] if item["what"] == what)


def test_the_gisaxs_procedure_cuts_at_yoneda_centres_and_averages_the_halves(galaxi_run) -> None:
    session, report = galaxi_run
    assert report["ok"] and report["procedure"] == "gisaxs" and report["measurement"] == "gisaxs"
    gisaxs = report["gisaxs"]
    assert gisaxs["cuts"]["horizontal_source"] == "yoneda"
    low, high = gisaxs["cuts"]["horizontal_rows"]
    assert low - 4 <= GALAXI["yoneda_row"] <= high + 4
    assert gisaxs["symmetry"]["x_px"] == pytest.approx(GALAXI["beam_center_px"][0], abs=1.0)
    assert report["geometry"]["beam_center_px"][0] == pytest.approx(gisaxs["symmetry"]["x_px"], abs=0.01)  # applied
    assert gisaxs["halves"]["side"] == "mean" and "averaged" in gisaxs["halves"]["reason"]
    assert _decision(report, "geometry")["decision"].startswith("from ")


def test_the_beam_centre_decision_states_the_centre_applied(galaxi_run) -> None:
    """The tool's answer is rounded to four significant digits (597.263 → 597.3); the decision and report.md
    say the centre that was applied, to 0.01 px, as the Results tab does."""
    _session, report = galaxi_run
    calibrated = report["gisaxs"]["symmetry"]["initial_x_px"]
    applied = report["geometry"]["beam_center_px"][0]
    assert _decision(report, "beam centre")["decision"] == (
        f"x {calibrated:.2f} → {applied:.2f} px ({applied - calibrated:+.2f} px)")
    assert f"symmetry axis x = {applied:.2f} px (calibrated {calibrated:.2f} px;" in pipeline_markdown(report)


def test_the_shoulder_gives_a_length_scale_that_is_never_called_the_spacing(galaxi_run) -> None:
    _session, report = galaxi_run
    spacing = report["gisaxs"]["spacing"]  # the record keeps its keys
    assert spacing["kind"] == "shoulder"  # a change of slope, not a resolved peak
    assert spacing["q"] == pytest.approx(SHOULDER_Q, rel=0.02)
    assert spacing["distance_nm"] == pytest.approx(2 * math.pi / spacing["q"] / 10)
    decision = _decision(report, "length scale")
    assert decision["decision"].startswith(f"2π/q* = {spacing['distance_nm']:.3g} nm at a shoulder of the cut")
    assert "not necessarily the particle spacing; check it with a model fit" in decision["why"]
    text = pipeline_markdown(report)
    section = text.split("## Length scale 2π/q*")[1].split("##")[0]
    assert "not necessarily the particle spacing; check it with a model fit" in section and "D ≈" not in text
    assert "In-plane spacing" not in text and "observed spacing" not in text
    steps = [step["summary"] for step in report["steps"] if step["tool"] == "in_plane_spacing"]
    assert steps and "2π/q* = " in steps[0] and "not necessarily the particle spacing" in steps[0]


def test_the_fit_reports_the_most_probable_and_asks_for_the_model_when_another_is_probable(galaxi_run) -> None:
    """Predict parameters needs no start (no D from 2π/q*); the run reports the most probable solution with its
    intervals, and asks for the model when another solution has a probability of at least 0.2."""
    session, report = galaxi_run
    call = session.catalog.fitter.calls[-1]
    assert call["components"] == () and 500 <= call["points"] <= 1000 and call["q_min"] >= 0  # the network chooses
    fit = report["gisaxs"]["fit"]
    assert [row["model"] for row in fit["solutions"]] == ["random_cylinder", "sphere"] and fit["chosen"] == 0
    assert [row["probability"] for row in fit["solutions"]] == [0.62, 0.38] and "distance_start_nm" not in fit
    assert fit["solutions"][1]["intervals"]["R_1"]["low68"] == pytest.approx(0.97 * 9.4)
    assert len(fit["curves"]) == 2 and len(fit["data"]["q_inv_angstrom"]) == call["points"]
    decision = _decision(report, "fit")
    assert decision["decision"] == "random_cylinder: R = 6.1 nm, D = 398 nm, χ²ᵣ = 1.22, probability 0.62"
    assert decision["why"].startswith("the most probable solution of Predict parameters on ")
    assert _decision(report, "fit intervals")["decision"] == (
        "R 6.10 [5.92, 6.28] nm, D 398 [358, 438] nm (refined value [68 % posterior interval])")
    assert any(item["what"] == "fit caveat" and "no interparticle correlation" in item["decision"] for item in report["decisions"])
    model = next(item for item in report["needs_attention"] if item["item"] == "model")
    assert model["why"] == (
        "2 solutions have a probability of at least 0.2 (rank 1, random_cylinder: R = 6.1 nm, D = 398 nm, χ²ᵣ = 1.22, "
        "probability 0.62; rank 2, sphere: R = 9.4 nm, D = 53.4 nm, χ²ᵣ = 1.31, probability 0.38): the curve alone "
        "does not decide the model.")
    assert "sphere" in model["hint"] and "53.4" in model["hint"]  # the solution that agrees with the spacing
    compact = compact_report(report)
    assert "data" not in compact["gisaxs"]["fit"] and "curves" not in compact["gisaxs"]["fit"]
    text = pipeline_markdown(report)
    assert text.startswith("# GISAXS") and "## Fit of I(qy)" in text and "Peaks (radial" not in text
    assert "| 1 (chosen) | random_cylinder | 0.62 | R 6.10 nm, σR/R 0.300, D 398.0 nm, σD/D 0.400 | 1.22 |  |" in text
    assert "Predict parameters: GIMaP's network proposes compositions" in text and "Huber" not in text


def test_a_close_second_solution_below_the_limit_is_not_a_question() -> None:
    from src.gimap.features.assistant.application.gisaxs_choice import ASK_PROBABILITY, probability, rivals

    assert ASK_PROBABILITY == 0.2
    solutions = [{"probability": 0.7}, {"probability": 0.2}, {"probability": 0.19}, {"probability": None}]
    assert rivals(solutions) == [solutions[1]] and rivals(solutions[:1]) == []
    assert rivals([{"chi2": 1.2}, {"chi2": 1.21}]) == []  # a record without probabilities: nothing to ask
    assert probability({"probability": "x"}) != probability({"probability": "x"})  # NaN


def test_the_command_line_writes_the_cuts_and_the_fit(galaxi_run, tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import write_outputs

    session, report = galaxi_run
    names = {path.name for path in write_outputs(session, report, tmp_path)}
    assert {"horizontal.csv", "vertical.csv", "fit_curve.csv", "fit_solutions.csv", "report.md"} <= names
    lines = (tmp_path / "fit_curve.csv").read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.startswith("#")]
    assert header[0].startswith("# GIMaP v")  # where the curve comes from, then its columns
    assert header[-1].startswith("# q (A^-1), I, sigma, I_fit")
    table = (tmp_path / "fit_solutions.csv").read_text(encoding="utf-8").splitlines()
    comments = [line for line in table if line.startswith("#")]
    assert comments[:-2] == header[:-1]  # the same lines as the curve tables: release, input with SHA-256, geometry
    assert comments[-2].startswith("# GISAXS fit solutions") and "in nm" in comments[-1]  # what it is, then the units
    assert table[len(comments)].startswith("rank,model,chi2") and len(table) == len(comments) + 3
    assert any(line.startswith("# input: ") and " sha256 " in line for line in comments)


def test_the_results_panel_shows_the_fit_and_saves_it(galaxi_run, tmp_path: Path) -> None:
    from src.gimap.features.assistant.presentation import GuidedResultsPanel, automatic_outcome

    _app()
    session, report = galaxi_run
    panel = GuidedResultsPanel(session.page.automation)
    refined = []
    panel.refineRequested.connect(lambda: refined.append(True))
    panel.show_report(report)
    section = panel.gisaxs
    assert section is not None and panel.peak_table is None
    assert section.fit_table.rowCount() == 2 and section.fit_plot.curve_count() == 2
    section.fit_table.selectRow(1)
    assert section.fit_plot.figure_state()["curves"][1][0] == "fit 2: sphere"
    assert section.fit_table.columnCount() == 6 and section.fit_table.item(0, 0).text() == "random cylinder"
    assert section.fit_table.horizontalHeaderItem(1).text() == "p" and section.fit_table.item(0, 1).text() == "0.62"
    assert section.fit_table.horizontalHeaderItem(5).text() == "χ²ᵣ"  # the reduced χ² Fitting shows for the solution
    assert section.fit_plot.log_x_check.isChecked()
    curve = Path(section.save_curve(str(tmp_path / "curve.csv")))
    table = Path(section.save_table(str(tmp_path / "table.csv")))
    assert curve.read_text(encoding="utf-8").startswith("# q (A^-1)") and "sphere" in table.read_text(encoding="utf-8")
    figure = Path(section.save_plot(str(tmp_path / "fit.png")))
    assert figure.stat().st_size > 5000
    section.findChild(QPushButton, "guidedRefineFit").click()
    assert refined == [True]
    # Fit details: closed until opened; every number of the selected solution, and Show in Fitting.
    assert section.fit_details is not None and not section.fit_details.is_expanded()
    section.fit_details.set_expanded(True)
    details = section.solution_details
    texts = [item.text() for item in details.values.findChildren(QLabel)]
    for expected in ("sphere: R", "9.4 nm (68 %: 9.12–9.68)", "σR/R", "Background", "Tail w", "0.01 nm⁻¹",
                     "probability", "0.38", "13.2 s"):
        assert expected in texts, (expected, texts)
    assert "evaluations" not in texts and "D started" not in details.settings.text()
    assert "the composition the network chose" in details.settings.text()
    assert "Predict parameters: the proposal network's" in details.settings.text()
    assert details.warnings.text() == "No warnings for this solution."
    shown = []
    panel.solutionRequested.connect(shown.append)
    details.show_button.click()
    row = shown[0]
    assert row["workflow"] == "native_v5" and row["combination"] == "sphere" and row["side"] == "mean"
    assert row["components"] == [{"type": "sphere", "weight": 1.0, "params": {"R": 9.4, "sigma_R": 0.3, "D": 53.4, "sigma_D": 0.4}}]
    assert len(row["native_q"]) == len(row["observed"]) == len(row["sigma"]) > 500
    section.fit_details.set_expanded(False)
    GuidedResultsPanel.details_open = False
    # The model choice is advice to look at (no field to fill in): a point to check, never a "question".
    state, line = automatic_outcome(report)
    assert state == "ok" and line.startswith("GISAXS — halves: mean of both halves, 2π/q* = ")
    assert line.endswith("; 1 point(s) to check.") and "question" not in line and "both_abs" not in line
    assert "most probable fit random cylinder R 6.1 nm (p 0.62, χ²ᵣ 1.22)" in line
    texts = [item.text() for item in panel.findChildren(QLabel)]
    assert texts[0].startswith("GISAXS — halves: mean of both halves") and not any("question" in text for text in texts)
    assert "1 point(s) to check — see below" in texts and any(text.startswith("Model choice: ") for text in texts)
    assert any(text.startswith("2 solutions have p ≥ 0.2: the curve alone does not decide the model.") for text in texts)
    # The decision on the line, its reason in the tooltip (2026-11 small-text review).
    halves = panel.findChild(QLabel, "guidedHalves")
    assert halves.text() == "Halves: Mean of both halves" and halves.toolTip() and " — " not in halves.text()
    model = next(label for label in panel.findChildren(QLabel) if label.text().startswith("Model choice: "))
    hint = model.toolTip()  # the hint: its details are said in the sections below
    assert hint.endswith("Choose the particle shape you expect and refine that model in Fitting; compare the fitted curves.")
    assert hint not in texts  # … not as a paragraph of its own
    assert not any(text.startswith("The cut is drawn on the image") for text in texts)  # (Sources' tooltip says it)
    panel.gisaxs.dispose()
    panel.deleteLater()


def _guided(session):
    from src.gimap.features.assistant.presentation import GuidedAnalysis

    written = []

    def save_text(path, text):
        written.append(path)
        Path(path).write_text(text, encoding="utf-8")
        return str(path)

    guided = GuidedAnalysis(session.page.automation, save_text=save_text)
    return guided, written


def test_results_follow_the_frame_shown_in_analyze(galaxi_run) -> None:
    import threading

    from src.gimap.features.assistant.presentation import GuidedResultsPanel

    _app()
    session, report = galaxi_run
    GuidedResultsPanel.details_open = False
    guided, _written = _guided(session)
    frame_a = report["frame"]
    frame_b = str(Path(frame_a).with_name("another_frame.tif"))
    sent = []
    guided.refineRequested.connect(lambda: sent.append("refine"))

    def buttons():
        results = guided.results
        return (results.findChild(QPushButton, "guidedRefineFit"), results.findChild(QPushButton, "guidedSaveReport"),
                results.gisaxs.solution_details.show_button)

    guided.frame_shown(frame_a)
    guided._finished(report)
    assert not guided.results.isHidden() and all(button.isEnabled() for button in buttons())
    guided.frame_shown(frame_a)  # the run's own re-reduction of the same file: nothing changes
    assert guided.report is report and all(button.isEnabled() for button in buttons())
    # Another file: the results of the first are put away, and nothing can send them on with the new frame.
    guided.frame_shown(frame_b)
    assert guided.results.isHidden() and guided.questions.isHidden() and guided.report is None
    assert not any(button.isEnabled() for button in buttons())
    assert guided.status_label.text() == "Results are for galaxi_data.tif; run again for this frame"
    assert guided.save_report() is None
    guided.results.refineRequested.emit()
    assert sent == []
    # A run on the second file; going back and forth brings each file's own results back.
    report_b = dict(report, frame=frame_b)
    guided._finished(report_b)
    guided.frame_shown(frame_a)
    assert guided.report is report and not guided.results.isHidden() and all(button.isEnabled() for button in buttons())
    assert guided.results.report is report and guided.status_label.text().startswith("Done — see the Results tab.")
    guided.frame_shown(frame_b)
    assert guided.report is report_b and guided.results.report is report_b
    guided.frame_shown(frame_a)
    # While a run lasts nothing changes; the file shown meanwhile is looked at when it ends.
    release = threading.Event()
    guided._thread = threading.Thread(target=release.wait, daemon=True)
    guided._thread.start()
    guided.frame_shown(frame_b)
    assert guided.report is report and all(button.isEnabled() for button in buttons())
    release.set()
    guided._thread.join()
    guided._done()
    guided._after_run()
    assert guided.report is report_b
    guided.results.gisaxs.dispose()
    guided.results.deleteLater()


def test_saves_are_proposed_next_to_the_data_and_say_where_they_went(galaxi_run, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    from src.gimap.app.presentation.components import visible_toasts
    from src.gimap.features.assistant.presentation import guided_text

    _app()
    session, report = galaxi_run
    monkeypatch.setattr(guided_text, "_SAVE_FOLDERS", {})
    proposed, answers = [], []

    def save_dialog(_parent, _title, start, _filters):
        proposed.append(start)
        return answers.pop(0), "CSV (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(save_dialog))
    guided, written = _guided(session)
    guided._finished(report)
    section = guided.results.gisaxs
    frame = Path(report["frame"])
    answers.append(str(tmp_path / "curve.csv"))
    assert section.save_curve() == str(tmp_path / "curve.csv")
    # Next to the data: its gimap_analysis folder when it exists (Analyze's exports), else the data folder.
    assert Path(proposed[-1]).parent in (frame.parent, frame.parent / "gimap_analysis")
    assert Path(proposed[-1]).name == "galaxi_data_gisaxs_fit_curve.csv"
    toast = visible_toasts(section.window())[-1]
    assert toast.property("level") == "ok" and toast.action_button.text() == "Open Folder"
    # The next save starts where the last one went (this session).
    answers.append(str(tmp_path / "table.csv"))
    section.save_table()
    assert Path(proposed[-1]) == tmp_path / "galaxi_data_gisaxs_fit_solutions.csv"
    answers.append(str(tmp_path / "fit.png"))
    assert section.save_plot() == str(tmp_path / "fit.png") and (tmp_path / "fit.png").stat().st_size > 5000
    assert Path(proposed[-1]).name == "galaxi_data_gisaxs_fit.png"
    # A file that cannot be written: an error toast, no exception (no error dialog).
    blocked = tmp_path / "blocked.csv"
    blocked.mkdir()
    answers.append(str(blocked))
    assert section.save_table() is None
    assert visible_toasts(section.window())[-1].property("level") == "error"
    answers.append(str(tmp_path / "galaxi_data_report.html"))
    assert guided.save_report() == str(tmp_path / "galaxi_data_report.html") and written
    assert Path(proposed[-1]).name == "galaxi_data_report.html"
    assert visible_toasts(guided.controls.window())[-1].action_button.text() == "Open Folder"

    def refuse(_path, _text):
        raise PermissionError(13, "Permission denied")

    guided._save_text = refuse
    answers.append(str(tmp_path / "again.html"))
    assert guided.save_report() is None
    toast = visible_toasts(guided.controls.window())[-1]
    assert toast.property("level") == "error" and "Permission denied" in toast.text()
    section.dispose()
    guided.results.deleteLater()


def test_the_saved_gisaxs_report_has_its_title_and_pictures(galaxi_run) -> None:
    import re

    from src.gimap.features.assistant.presentation.guided_report import report_page

    _app()
    session, report = galaxi_run
    page = report_page(report, None, session.page.automation)
    assert "<h1>GISAXS report</h1>" in page and "GIWAXS" not in page.split("<h1>")[1].split("</p>")[0]
    assert "Automatic analysis, no AI" in page and page.count("data:image/png;base64,") >= 2
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", page)
    assert len(captions) == 2 and not any("ring" in caption.lower() for caption in captions), captions
    assert "most probable solution of Predict parameters" in captions[1]


def test_the_results_built_in_chinese_are_in_chinese(galaxi_run) -> None:
    # The results are built after the window was translated: they are translated themselves.
    from PyQt5.QtWidgets import QToolButton

    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.assistant.presentation import GuidedResultsPanel

    _app()
    session, report = galaxi_run
    apply_language("zh")
    try:
        panel = GuidedResultsPanel(session.page.automation)
        panel.show_report(report)
        assert panel.findChild(QToolButton, "guidedSaveFitPlot").text() == "保存图…"
        assert "拟合" in panel.findChild(QPushButton, "guidedRefineFit").text()
        panel.gisaxs.dispose()
        panel.deleteLater()
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_halves_and_spacing_rules_on_synthetic_curves() -> None:
    q = np.linspace(-0.2, 0.2, 401)
    intensity = 100 * np.exp(-np.abs(q) / 0.05) + 5
    pixels = np.full(q.shape, 10.0)
    assert choose_halves(q, intensity, pixels).side == "mean"
    gappy = pixels.copy()
    gappy[q < -0.01] = 0  # the qy < 0 half behind a module gap
    shadowed = intensity.copy()
    shadowed[q < -0.01] = np.nan
    choice = choose_halves(q, shadowed, gappy)
    assert choice.side == "positive" and choice.fit_side == "positive"

    class Peak:
        def __init__(self, q, snr, flags=()):
            self.q, self.snr, self.flags = q, snr, tuple(flags)

    assert spacing_from_peaks([Peak(0.02, 30, ("broad",)), Peak(0.05, 8)], q_min=0.005).kind == "maximum"
    shoulder = spacing_from_peaks([Peak(0.0117, 120, ("broad", "fit_failed")), Peak(0.001, 50)], q_min=0.005)
    assert shoulder.kind == "shoulder" and shoulder.distance_nm == pytest.approx(2 * math.pi / 0.0117 / 10)
    assert spacing_from_peaks([Peak(0.02, 3, ("broad",))], q_min=0.005) is None


def test_ask_ai_offers_the_gisaxs_results_for_a_gisaxs_frame() -> None:
    from src.gimap.features.assistant.application import GISAXS_GOALS, SYSTEM_PROMPT, task_message
    from src.gimap.features.assistant.presentation.start_dialog import AssistantStartDialog
    from src.gimap.integrations.state import InMemorySettingsRepository

    _app()
    status = {"file": "galaxi_data.tif", "measurement": "gisaxs", "geometry": {"distance_mm": 1730.0}}
    dialog = AssistantStartDialog(InMemorySettingsRepository({}), status=status)
    assert tuple(dialog.goal_checks) == GISAXS_GOALS and dialog.ring_spin.isHidden()
    goals = dialog.goals()
    assert goals.goals == GISAXS_GOALS and goals.ring_q is None
    assert task_message(goals, status).startswith("Analyse the GISAXS frame")
    assert "refine_beam_center_symmetry" in SYSTEM_PROMPT and "shoulder" in SYSTEM_PROMPT
    dialog.deleteLater()


def test_a_solution_opens_in_fitting_with_its_parameters(galaxi_run) -> None:
    """Show in Fitting hands Fitting a ``native_v5`` candidate it converts into its own parameters."""
    from src.gimap.features.assistant.presentation.guided_details import candidate_row
    from src.gimap.features.fitting.domain.native_solution import native_solution_mapping

    _session, report = galaxi_run
    fit = report["gisaxs"]["fit"]
    converted = native_solution_mapping(candidate_row(fit, 1)).mapping  # the sphere solution of the fake fitter
    sphere = converted.components[0]
    assert sphere.shape == "Sphere" and sphere.parameters["radius"] == 9.4
    assert sphere.parameters["sigma_radius"] == pytest.approx(9.4 * 0.3)  # σR/R 0.3 → nm
    assert sphere.parameters["sigma_diameter"] == pytest.approx(53.4 * 0.4)
    assert set(converted.global_parameters) == {"background", "sigma_res", "nu_res", "int_res", "k_value"}
    assert candidate_row({"solutions": fit["solutions"]}, 0) is None  # an older report without the points


def test_the_halves_the_analysis_chooses_are_for_its_frame_not_the_stored_choice(galaxi_run) -> None:
    """The automatic analysis sets the halves through ``AnalyzeAutomation.set_halves``. What it chooses for one
    sample (a half in the shadow) used to be written to the settings, and was the hand-over of the next sample
    and the next session."""
    _app()
    session, _report = galaxi_run
    page = session.page
    model = page.view_model
    stored = model._read_setting("analyze", "fit_side", "both_abs")
    shown = model.fit_side
    other = "negative" if shown != "negative" else "positive"
    done = []
    try:
        page.automation().set_halves(other, lambda ok, text: done.append((ok, text)))
        assert done == [(True, f"halves: {other}")] and model.fit_side == other
        assert page.halves_combo.currentData() == other  # the Cuts step shows the halves of this frame
        assert model._read_setting("analyze", "fit_side", "both_abs") == stored  # … and the stored choice is as it was
        model.forget_frame_halves()  # another file, or Clear
        assert model.fit_side == (stored if stored in ("both_abs", "mean", "negative", "positive") else "both_abs")
    finally:
        model.set_fit_side(shown, remember=False)
        page._sync_fit_side()
        page._sync_halves()


# -- solutions at a bound of their range (the referees of the JAC draft: a solution on GALAXI had D at its bound) ----


@pytest.fixture(scope="module")
def bounded_runs():
    from src.gimap.app.headless_assistant import open_headless_session

    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    options = PipelineOptions(technique="gisaxs", geometry=_geometry(), incidence_deg=GALAXI["incidence_deg"])
    session.catalog.fitter = BoundFitter()
    report = session.run_pipeline(options)
    yield session, report
    session.close()


def test_the_most_probable_solution_is_reported_and_its_values_at_a_bound_are_a_point_to_check(bounded_runs) -> None:
    """The run reports the most probable solution even with a value at a bound of its range; that value is a point
    to check, and the tables name every value at a bound. Every number is formatted once, from the fit record at
    full precision: 48.3484 is 48.3 in the table (it was 48.4 from the tool's answer rounded to four digits first)."""
    from src.gimap.features.assistant.application import fit_solutions_csv
    from src.gimap.features.assistant.application.gisaxs_report import gisaxs_batch_row

    _session, report = bounded_runs
    fit = report["gisaxs"]["fit"]
    assert fit["chosen"] == 0 and fit["solutions"][0]["model"] == "random_cylinder"
    # the fit's own warning for D at its bound is said once, by at_bounds (which bound); the others pass on
    assert fit["solutions"][0]["warnings"] == [UNCONSTRAINED]
    decision = _decision(report, "fit")
    assert decision["decision"] == "random_cylinder: R = 8.6 nm, D = 500 nm, χ²ᵣ = 1.26, probability 0.55"
    assert any(item["what"] == "fit caveat" and item["decision"] == "D at its upper bound, σh/h at its upper bound"
               for item in report["decisions"])
    assert any(item["what"] == "fit caveat" and item["decision"] == UNCONSTRAINED for item in report["decisions"])
    point = next(item for item in report["needs_attention"] if item["item"] == "fit")
    assert point["why"] == ("The reported solution (random_cylinder) has D at its upper bound, σh/h at its upper "
                            "bound: those values are not determined by the curve.")
    model = next(item for item in report["needs_attention"] if item["item"] == "model")
    assert model["why"].startswith("2 solutions have a probability of at least 0.2 (rank 1, random_cylinder: R = 8.6 nm")
    assert "rank 2, random_cylinder: R = 6.01 nm, D = 30.9 nm, χ²ᵣ = 1.28, probability 0.25" in model["why"]
    assert "rank 3" not in model["why"]  # p 0.15: below the limit
    text = pipeline_markdown(report)
    assert f"| D at its upper bound, σh/h at its upper bound; {UNCONSTRAINED} |" in text
    # One number of decimals per quantity in every row: R 8.60 next to R 9.44, σR/R 0.300 (not 0.3).
    assert "| 3 | sphere | 0.15 | R 9.44 nm, σR/R 0.300, D 48.3 nm, σD/D 0.400 | 1.36 |  |" in text
    assert "| 1 (chosen) | random_cylinder | 0.55 | R 8.60 nm, " in text and "D 500.0 nm" in text
    assert gisaxs_batch_row(report)[2] == (
        "random_cylinder R 8.60 nm, D 500.0 nm, χ²ᵣ 1.26, p 0.55 ⚠ D at its upper bound, σh/h at its upper bound")
    table = fit_solutions_csv(fit).splitlines()
    assert table[2].startswith("rank,model,chi2,probability,") and table[2].endswith(",at_bounds,chosen")
    assert "R_low68,R_high68,h_low68,h_high68,D_low68,D_high68" in table[2]
    assert table[3].startswith("1,random_cylinder,1.26,0.55,") and table[3].endswith(",D:upper sigma_h:upper,True")
    assert table[5].startswith("3,sphere,") and table[5].endswith(",,False")


def test_the_results_tab_selects_the_reported_solution_and_names_the_bounds(bounded_runs) -> None:
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.assistant.presentation import GuidedResultsPanel, automatic_outcome
    from src.gimap.features.assistant.presentation.guided_words import words

    _app()
    session, report = bounded_runs
    panel = GuidedResultsPanel(session.page.automation)
    panel.show_report(report)
    section = panel.gisaxs
    assert section.fit_table.currentRow() == 0  # the solution the run reports, drawn on the plot
    assert section.fit_table.item(0, 0).text() == "random cylinder ⚠" and section.fit_table.item(0, 4).text() == "500 ⚠"
    assert section.fit_table.item(2, 0).text() == "sphere" and section.fit_table.item(2, 4).text() == "48.3"
    assert [section.fit_table.item(row, 1).text() for row in range(4)] == ["0.55", "0.25", "0.15", "0.05"]
    bounds = section.findChild(QLabel, "guidedFitBounds")  # values the curve does not determine: a warning
    assert bounds.text() == "⚠ Not determined by the curve (at a search bound) — 1: D, σh/h; 2: σh/h; 4: Tail ν"
    assert bounds.property("gimapRole") == "warning"
    assert bounds.toolTip() == ("Values at a search bound, not determined by the curve — 1: D at its upper bound, σh/h at "
                                "its upper bound; 2: σh/h at its lower bound; 4: Tail ν at its upper bound.")
    texts = [item.text() for item in panel.findChildren(QLabel)]
    assert not any(text.startswith("Reported: rank") for text in texts), texts
    assert any(text.startswith("2 solutions have p ≥ 0.2") for text in texts)
    assert any(text.startswith("2π/q* = ") and text.endswith("not necessarily the particle spacing.") for text in texts)
    fitted = next(text for text in texts if text.startswith("Fitted: "))
    assert fitted.endswith(" · R, h and D in nm.")  # (what the fits are: the tooltip of “Fit of I(qy)”)
    heading = next(label for label in panel.findChildren(QLabel) if label.text() == "Fit of I(qy)")
    assert heading.toolTip().startswith("Predict parameters: GIMaP's network proposes the particles")
    assert "⚠ D at its upper bound" in section.solution_details.warnings.text()
    assert ("most probable fit random cylinder R 8.6 nm (p 0.55, χ²ᵣ 1.26) ⚠ D at its upper bound, σh/h at its upper "
            "bound") in automatic_outcome(report)[1]
    panel.gisaxs.dispose()
    panel.deleteLater()
    point = next(item for item in report["needs_attention"] if item["item"] == "fit")
    model = next(item for item in report["needs_attention"] if item["item"] == "model")
    apply_language("zh")
    try:  # by its form; the values at a bound as a list, the warnings of Predict parameters too
        assert words(point["why"]) == "报告的解（random_cylinder）：D 处于搜索上界、σh/h 处于搜索上界；这些数值不由曲线确定。"
        assert words(model["why"]).startswith("2 个解的概率都不低于 0.2（rank 1, random_cylinder: R = 8.6 nm")
        assert words(model["hint"]) == "请选择你预期的粒子形状，在拟合中精修这个模型，并比较拟合曲线。"
        assert words(UNCONSTRAINED) == "曲线对 σh/h 约束很少（其 68 % 区间覆盖了允许范围的大部分）"
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_the_gisaxs_report_says_what_its_numbers_are(galaxi_run) -> None:
    """As the GIWAXS report does: q is kinematic — for GISAXS refraction moves q_z, the Yoneda band's most, by an
    amount of the order of k·sin αi — and the σ are statistical only, in the frame's error model."""
    _session, report = galaxi_run
    q = _decision(report, "q")
    shift = 2 * math.pi / GALAXI["wavelength_angstrom"] * math.sin(math.radians(GALAXI["incidence_deg"]))
    assert q["decision"] == "kinematic (no refraction correction)"
    assert "that of the Yoneda band (αf ≈ αc) most" in q["why"]
    assert f"k·sin αi = {shift:.3g} Å⁻¹ here (αi = {GALAXI['incidence_deg']:g}°" in q["why"]
    assert q["why"].endswith("q_y and the length scale 2π/q* of the horizontal cut barely change")
    errors = _decision(report, "uncertainties")
    assert errors["decision"] == "statistical only" and report["error_model"] == "poisson"
    assert "Poisson counting statistics of each bin" in errors["why"]
    assert "the intervals of R, h and D are the posterior of Predict parameters" in errors["why"]
    whats = [item["what"] for item in report["decisions"]]
    assert whats.index("q") < whats.index("uncertainties") < whats.index("horizontal cut")


def test_the_length_scale_decision_says_what_it_is_once(galaxi_run) -> None:
    _session, report = galaxi_run
    decision = _decision(report, "length scale")
    assert decision["why"] == ("a length scale, not necessarily the particle spacing; check it with a model fit; a form "
                               "factor or the beam tail makes a shoulder too")
    line = next(text for text in pipeline_markdown(report).splitlines() if text.startswith("- **length scale**"))
    assert line.count("a length scale") == 1 and line.count("shoulder of the cut") == 1 and line.count("2π/q*") == 1


def test_a_linear_term_at_zero_is_said_and_not_counted_as_a_search_bound(galaxi_run) -> None:
    """The amplitudes, BG and the beam-tail amplitude A are solved non-negative: one at 0 is a caveat of the reported
    solution, not a value at a search bound (the GALAXI sphere the run reports has BG = 0)."""
    from src.gimap.features.assistant.application.gisaxs_choice import zero_terms

    _session, report = galaxi_run
    fit = report["gisaxs"]["fit"]
    assert fit["chosen"] == 0 and fit["solutions"][0]["globals"]["resolution_amplitude"] == 0.0
    caveat = next(item for item in report["decisions"] if item["what"] == "fit caveat" and "non-negativity" in item["decision"])
    assert caveat["decision"] == "Tail A at 0, the non-negativity limit"
    assert "not counted among the values at a bound of their range" in caveat["why"]
    assert caveat["why"].endswith("Tail w and Tail ν do not change the curve")
    assert "A non-linear value at a bound of its range (the note column)" in pipeline_markdown(report)
    sphere = {"chi2": 1.36, "converged": True, "at_bounds": [], "globals": {"background": 0.0, "resolution_amplitude": 252.5}}
    assert zero_terms(sphere) == ["Background"]
    # the fit's own warning for a linear term at its limit is not said twice (the caveat above says it)
    from src.gimap.features.assistant.application.gisaxs_tools import _solution

    row = _row([0.01, 0.02], [1.0, 2.0], [0.1, 0.1], 1, "sphere", 5.0, 40.0, 1.1,
               warnings=["res_amplitude stopped at a bound of its range", "R_1 stopped at a bound of its range"])
    assert _solution(row)["warnings"] == ["R_1 stopped at a bound of its range"]  # at_bounds empty: kept


class PoorFitter(FakeFitter):
    """One converged solution far from its curve: χ²ᵣ of hundreds, as a fit of the public AgBh SAXS frame gives
    (the Bragg rings of a calibrant, no particles)."""

    def __call__(self, q, intensity, sigma, *, components=(), report=None, cancelled=None, cut_qz=None):
        self.calls.append({"points": len(q), "components": components, "q_min": float(np.min(q))})
        return [_row(q, intensity, sigma, 1, "sphere", 7.18, 16.9, 472.0)]


def test_a_reported_fit_far_from_its_curve_is_a_point_to_check() -> None:
    """Fitting calls a σ-weighted fit with χ²ᵣ above 10 a poor fit; the report uses the same limit for χ²ᵣ."""
    import tools.gimap_agent as agent
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.assistant.application.gisaxs_choice import POOR_CHI2, poor_fit
    from src.gimap.features.assistant.presentation.guided_words import words
    from src.gimap.features.fitting.application.series_fit import POOR_CHI2 as FITTING_POOR_CHI2

    assert POOR_CHI2 == FITTING_POOR_CHI2
    assert not poor_fit({"chi2": 472.0, "converged": False}) and not poor_fit({"chi2": 9.9, "converged": True})
    session = open_headless_session(DATA / GALAXI["frame"], saved_profiles=False)
    try:
        session.catalog.fitter = PoorFitter()
        report = session.run_pipeline(PipelineOptions(technique="gisaxs", geometry=_geometry(),
                                                      incidence_deg=GALAXI["incidence_deg"]))
    finally:
        session.close()
    point = next(item for item in report["needs_attention"] if item["item"] == "fit")
    assert point["why"].startswith("The reported solution (sphere) has χ²ᵣ = 472, above 10: it misses the curve")
    assert "its R and D are not measurements" in point["why"] and point["option"] is None
    assert agent.exit_code([report]) == 2
    apply_language("zh")
    try:
        assert "χ²ᵣ = 472，大于 10" in words(point["why"]) and words(point["hint"]).startswith("请对照曲线查看拟合")
    finally:
        apply_language(DEFAULT_LANGUAGE)
