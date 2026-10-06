"""GISAXS in the automatic analysis: Yoneda cut, symmetry, halves, spacing, fit, report and results panel.

The real frame is BornAgain's GALAXI example (tests/data/external): silver spheres in a film, a
radial paracrystal with a peak distance of 53.6 nm in BornAgain's model of it.
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
BORNAGAIN_DISTANCE_NM = 53.6


def _geometry() -> dict:
    return {
        "distance_mm": GALAXI["distance_m"] * 1e3, "beam_center_px": GALAXI["beam_center_px"],
        "wavelength_angstrom": GALAXI["wavelength_angstrom"], "energy_kev": 12.398419843 / GALAXI["wavelength_angstrom"],
        "pixel_size_um": [GALAXI["pixel_size_m"] * 1e6] * 2, "source_image": "galaxi_geometry.json",
        "shape": [1043, 981], "from_file": True,
    }


class FakeFitter:
    """Two solutions of different families with close χ²; records what it was asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, intensity, sigma, *, components=(), distance_nm=None, report=None, cancelled=None):
        self.calls.append({"points": len(q), "distance_nm": distance_nm, "q_min": float(np.min(q))})
        grid = np.linspace(0.05, 2.5, 50)

        def row(rank, model, radius, distance, chi2):
            native = 10.0 * np.asarray(q, dtype=float)  # nm⁻¹, as Fitting's quick fit gives it
            return {
                "rank": rank, "combination": model, "best_chi2_weighted": chi2, "best_log_rmse": 0.15, "converged": True,
                "warnings": [], "display_q": grid.tolist(), "display_fit": (1e3 / (1 + (grid * radius) ** 4)).tolist(),
                "components": [{"type": model, "weight": 1.0, "amplitude": 2.5e4,
                                "params": {"R": radius, "sigma_R": 0.3, "D": distance, "sigma_D": 0.4}}],
                "global_params": {"background": 0.8, "resolution_amplitude": 0.0, "sigma_Res": 0.01, "nu_Res": 1.0},
                "nfev": 140, "seconds": 2.34, "conditions_source": "single_component_screen",
                "algorithm": "Bounded soft-L1 least_squares with weighted nonnegative linear-amplitude least squares",
                "native_q": native.tolist(), "native_fit": (1e3 / (1 + (native * radius) ** 4)).tolist(),
                "observed": list(map(float, intensity)), "sigma": list(map(float, sigma)),
            }

        return [row(1, "random_cylinder", 6.1, 398.0, 1.22), row(2, "sphere", 9.4, 53.4, 1.31)]


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


def test_the_spacing_is_a_shoulder_at_the_distance_of_bornagains_model(galaxi_run) -> None:
    _session, report = galaxi_run
    spacing = report["gisaxs"]["spacing"]
    assert spacing["kind"] == "shoulder"  # a change of slope, reported as a hint
    assert spacing["distance_nm"] == pytest.approx(BORNAGAIN_DISTANCE_NM, rel=0.05)
    assert "hint" in _decision(report, "spacing")["why"]


def test_the_fit_starts_from_the_spacing_and_asks_for_the_model_when_families_tie(galaxi_run) -> None:
    session, report = galaxi_run
    call = session.catalog.fitter.calls[-1]
    assert call["distance_nm"] == pytest.approx(report["gisaxs"]["spacing"]["distance_nm"])
    assert 500 <= call["points"] <= 1000 and call["q_min"] >= 0
    fit = report["gisaxs"]["fit"]
    assert [row["model"] for row in fit["solutions"]] == ["random_cylinder", "sphere"]
    assert len(fit["curves"]) == 2 and len(fit["data"]["q_inv_angstrom"]) == call["points"]
    assert any(item["what"] == "fit caveat" and "no interparticle correlation" in item["decision"] for item in report["decisions"])
    model = next(item for item in report["needs_attention"] if item["item"] == "model")
    assert "sphere" in model["hint"] and "53.4" in model["hint"]  # the solution that agrees with the spacing
    compact = compact_report(report)
    assert "data" not in compact["gisaxs"]["fit"] and "curves" not in compact["gisaxs"]["fit"]
    text = pipeline_markdown(report)
    assert text.startswith("# GISAXS") and "## 拟合 / Fit of I(qy)" in text and "Peaks (radial" not in text


def test_the_command_line_writes_the_cuts_and_the_fit(galaxi_run, tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import write_outputs

    session, report = galaxi_run
    names = {path.name for path in write_outputs(session, report, tmp_path)}
    assert {"horizontal.csv", "vertical.csv", "fit_curve.csv", "fit_solutions.csv", "report.md"} <= names
    header = (tmp_path / "fit_curve.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("# q (A^-1), I, sigma, I_fit")
    table = (tmp_path / "fit_solutions.csv").read_text(encoding="utf-8").splitlines()
    assert table[0].startswith("rank,model,chi2") and len(table) == 3


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
    assert section.fit_table.columnCount() == 5 and section.fit_table.item(0, 0).text() == "random cylinder"
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
    for expected in ("sphere: R", "9.4 nm", "σR / R", "constant background", "σ_Res", "0.01 nm⁻¹", "140", "2.3 s"):
        assert expected in texts, (expected, texts)
    assert "D started at the in-plane spacing" in details.settings.text() and "soft-L1" in details.settings.text()
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
    assert state == "ok" and line.startswith("GISAXS — halves: mean of both halves, D ≈ ")
    assert line.endswith("; 1 point(s) to check.") and "question" not in line and "both_abs" not in line
    texts = [item.text() for item in panel.findChildren(QLabel)]
    assert texts[0].startswith("GISAXS — halves: mean of both halves") and not any("question" in text for text in texts)
    assert "1 point(s) to check — see below" in texts and any(text.startswith("Model choice: ") for text in texts)
    assert any(text.startswith("Halves: Mean of both halves — ") for text in texts)
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
    assert "best fit" in captions[1]


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
