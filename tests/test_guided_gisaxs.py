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
    state, line = automatic_outcome(report)
    assert state == "ok" and "halves: mean" in line and "question" in line
    panel.gisaxs.dispose()
    panel.deleteLater()


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
