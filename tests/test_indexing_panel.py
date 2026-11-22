"""Analyze's Phases window and the phase steps of the automatic GIWAXS analysis.

- phases given to the page draw their reflections on the detector, the q map and I(q); the window lists the phases
  and the reflections, fits the pole figure of a reflection's ring and exports it as CSV with a JSON record;
- every text of the window has its Chinese;
- on the public P08 frame the automatic analysis with ``phases`` assigns the rings (a decision line, the peak rows,
  a report section) and fits the pole figure of each analysed ring next to its Hermans factor.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.app.presentation.i18n.zh_indexing import INDEXING_ZH
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_crystal_indexing import MAPI_CIF, PBI2_CIF
from tests.test_review_analyze_run_and_clear import _frames, _profiles

ANALYZE = Path(__file__).resolve().parents[1] / "src/gimap/features/analyze"
P08 = Path(__file__).parent / "data" / "external" / "giwaxs_p08_mapi"
FRAME, PONI = "S121_MAI_A2_00841.tif", "LaB6_2021_12_DESY_P08.poni"


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_phases_are_drawn_listed_fitted_and_exported(tmp_path: Path, monkeypatch) -> None:
    apply_language("en")
    paths = _frames(tmp_path, 1)  # a synthetic GIWAXS frame with a ring at q = 1.1 Å⁻¹
    (tmp_path / "mapi.cif").write_text(MAPI_CIF, encoding="utf-8")
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(paths[0])])
        _done(page)
        assert page.reflection_count() == 0  # no phases: nothing drawn
        automation = page.automation()
        result = automation.set_phases([
            {"name": "cubic", "cell": (9.8935, 9.8935, 9.8935), "centring": "F", "orientation": "plane", "indices": (1, 1, 1)},
            str(tmp_path / "mapi.cif"),
        ])
        assert [phase["name"] for phase in result["phases"]] == ["cubic", "MAPbI3 (tetragonal)"]
        on_detector = page.reflection_count()
        assert on_detector > 0 and page._curve_reflections.shown > 0  # spots on the image, ring lines on I(q)
        for view in (1, 2):  # the q map, then the cake (made on the task runner)
            page.set_view(view)
            _done(page)
            assert page.reflection_count() > 0, view
        page.set_view(1)
        _done(page)
        found = automation.assign_peaks([{"q": 1.1, "q_err": 0.001, "fwhm": 0.04}])
        assert found["assignments"][0]["phase"] == "cubic" and found["assignments"][0]["hkl"] == "111"  # at 1.1000 Å⁻¹
        panel = page.open_indexing()
        assert panel.phase_table.rowCount() == 2 and panel.reflection_table.rowCount() > 10
        assert page.phases_button.isChecked()
        # Hide the second phase: its reflections leave the table and the image.
        panel.phase_table.cellWidget(1, 0).setChecked(False)
        assert all(panel.reflection_table.item(row, 0).text() == "cubic" for row in range(panel.reflection_table.rowCount()))
        # The pole figure of the ring at 1.1 Å⁻¹ (an isotropic ring: almost no oriented fraction).
        page.view_model.set_chi_window(1.07, 1.13)
        page.run_analysis()
        _done(page)
        panel.model_combo.setCurrentIndex(panel.model_combo.findData("gaussian_mosaic"))
        panel.fit_pole()
        _done(page)
        pole = page.indexing.last_pole
        assert pole is not None and pole.fit.ok and pole.fit.values["oriented_fraction"] < 0.2
        assert "oriented fraction" in panel.result_label.text() and panel.pole_plot.curve_count() >= 2
        reference = pole.fit.herman_isotropic
        result_text = panel.result_label.text()
        measured = next(line for line in result_text.splitlines() if "Measured f =" in line)
        assert f"Isotropic reference f_iso = {reference:#.3g}" in measured
        for end in ("min", "max"):
            assert pole.systematics["range_sources"][end]["scenario"] in result_text
        target = tmp_path / "out" / "film_pole_figure.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kw: (str(target), "")))
        panel.export_button.click()
        text = target.read_text(encoding="utf-8")
        assert "# columns: chi_deg  N  sigma  measured  model" in text and len(text.splitlines()) > 90
        summary = next(line for line in text.splitlines() if line.startswith("# pole figure summary:"))
        assert "measured f =" in summary
        assert float(summary.split("f_iso = ")[1].replace("−", "-")) == pytest.approx(reference, rel=.005)
        record = json.loads((tmp_path / "out" / "film_indexing.json").read_text(encoding="utf-8"))
        assert record["pole_figure"]["model"] == "gaussian_mosaic"
        assert record["pole_figure"]["herman_isotropic"] == reference
        csv_record = next(line.removeprefix("# pole figure: ") for line in text.splitlines()
                          if line.startswith("# pole figure: "))
        assert json.loads(csv_record)["herman_isotropic"] == reference
        assert record["pole_figure"]["values"]["oriented_fraction"] == pytest.approx(pole.fit.values["oriented_fraction"])
        assert [phase["name"] for phase in record["indexing"]["phases"]] == ["cubic", "MAPbI3 (tetragonal)"]
        assert record["analysis"]["geometry"]
        apply_language("zh", [panel])
        assert panel.windowTitle() == "物相与指标化" and panel.fit_button.text() == "拟合极图"
        assert f"各向同性参考值 f_iso = {reference:#.3g}" in panel.result_label.text()
        for end in ("min", "max"):
            assert ZH[pole.systematics["range_sources"][end]["scenario"]] in panel.result_label.text()
        panel.refresh()
        assert panel.phase_table.item(0, 4).text() == "(1 1 1) 平行于衬底"
    finally:
        apply_language("en")
        _close(page)


def test_every_text_of_the_phases_window_has_its_chinese() -> None:
    sources = [ANALYZE / name for name in (
        "presentation/views/indexing_view.py", "presentation/indexing_panel.py", "presentation/indexing_texts.py",
        "presentation/bindings/indexing.py", "application/pole_figure.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for english, chinese in INDEXING_ZH.items():
        assert ZH[english] == chinese, english
        assert english.split("{")[0][:28] in text, english


def test_assignment_and_pole_fit_warnings_are_visible_in_both_languages():
    from src.gimap.features.analyze.application.pole_figure import PoleFigure
    from src.gimap.features.analyze.domain.crystal.pole_figure import PoleFit
    from src.gimap.features.analyze.presentation.indexing_texts import pole_result_text
    from src.gimap.features.assistant.presentation.guided_tables import assignment_cells

    fit = PoleFit(True, "gaussian", values={"fwhm_deg": 120.}, chi2_red=6.2, at_bound=("fraction_iso",),
                  converged=True)
    result = PoleFigure(fit, (0.98, 1.02), 1.2, "given", False, 0.0)
    peak = {"phase": "A", "hkl": "002", "ambiguous": True, "delta_q": -0.006, "relative_d": 0.006}
    try:
        for language, label, unmatched, headline, reasons in (
            ("en", "002 (ambiguous)", "Unassigned", "⚠ Pole fit not determined",
             ("reduced χ² exceeds 5", "FWHM exceeds 100°", "isotropic fraction at a bound")),
            ("zh", "002（多种可能）", "未指认", "⚠ 极图拟合未能确定取向",
             ("约化 χ² 大于 5", "半高宽大于 100°", "各向同性分数达到边界")),
        ):
            apply_language(language)
            cells = assignment_cells(peak)
            assert cells[1][0] == label and "Δd/d = +0.600%" in cells[2][1]
            assert assignment_cells({"phase": None})[0] == unmatched
            first = pole_result_text(result).splitlines()[0]
            assert first.startswith(headline) and all(reason in first for reason in reasons)
    finally:
        apply_language("en")


@pytest.mark.skipif(not (P08 / FRAME).is_file(), reason="the public P08 frame is not in this checkout")
def test_the_automatic_analysis_assigns_the_rings_and_fits_their_pole_figures(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.assistant.application import PipelineOptions, pipeline_markdown

    data = tmp_path / "data"
    data.mkdir()
    for name in (FRAME, PONI):
        shutil.copy2(P08 / name, data / name)
    (tmp_path / "mapi.cif").write_text(MAPI_CIF, encoding="utf-8")
    (tmp_path / "pbi2.cif").write_text(PBI2_CIF, encoding="utf-8")
    session = open_headless_session(data / FRAME, saved_profiles=False, out_dir=tmp_path / "out")
    try:
        report = session.run_pipeline(PipelineOptions(
            calibration=str(data / PONI), incidence_deg=0.075, follow_detection=True,
            phases=(str(tmp_path / "mapi.cif"), str(tmp_path / "pbi2.cif")),
        ))
        decision = next(item for item in report["decisions"] if item["what"] == "phase assignment")
        assert "MAPbI3 (tetragonal) 110" in decision["decision"] and "PbI2 001" in decision["decision"]
        assert "max(3 σ(q0), FWHM/2)" in decision["why"]
        rows = {round(row["q"], 2): row for row in report["peaks"]}
        assert rows[1.0]["hkl"] == "110" and abs(rows[1.0]["delta_q"]) < 0.006
        assert rows[1.0]["ambiguous"] and any(alt["hkl"] == "002" for alt in rows[1.0]["alternatives"])
        assert rows[1.0]["relative_d"] == pytest.approx(rows[1.0]["q_hkl"] / rows[1.0]["q"] - 1)
        assert report["phase_assignment"]["q_axis"]["kind"] == "kinematic"
        assert report["phase_assignment"]["phases"][1]["name"] == "PbI2"
        assert any(item["what"] == "pole figure" for item in report["decisions"])
        reliable = [peak for peak in report["peaks"] if not peak["caveat"] and "at_edge" not in peak["flags"]]
        largest = max(reliable, key=lambda peak: peak["fit"]["area"])
        marked = [ring["q"] for ring in report["rings"] if ring.get("strongest_by_area")]
        assert marked == pytest.approx([largest["q"]])
        for ring in report["rings"]:  # next to the Hermans factor of every analysed ring
            pole = ring["pole_figure"]
            assert ring.get("herman") is not None and pole["ok"]
            assert 0.0 <= pole["oriented_fraction"] <= 1.0 and pole["oriented_fraction_err"] >= 0.0
        markdown = pipeline_markdown(report)
        assert "## Phase assignment" in markdown and "pole figure (Gaussian + isotropic)" in markdown
        assert "## Crystal orientation" in markdown and "measured ring f" in markdown
        from src.gimap.features.assistant.application.pipeline_report import compact_report

        ring = compact_report(report)["crystal_orientation"][0]["rings"][0]
        assert "fitted" not in ring and "values" not in ring["components"][0]
        assert ring["components"][0]["hkl"] and ring["effective_dof"] > 0
        assert "Phases: the nearest reflection of the phases given" in markdown
        assert "ambiguous" in markdown and "Δd/d =" in markdown and "Å⁻¹ unassigned" in markdown
        assert session.page.reflection_count() > 0  # the phases given are drawn in Analyze too
    finally:
        session.close()
