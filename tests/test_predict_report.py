"""Reporting of existing Predict results without changing the fit or its reported intervals."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.fitting.application import solution_export
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.posterior import refine, reporting
from src.gimap.features.fitting.presentation.single import predict_texts, project
from src.gimap.features.fitting.presentation.single.session import Solution
from src.gimap.shared import fit_names


def test_grazing_limits_depend_on_data_level_and_shape():
    sphere = fm.FitModel((fm.new_component("sphere", radius=1),))
    flags = reporting.grazing_flags
    assert flags(sphere, 1.2, 1, True) == []
    assert flags(sphere, 1.3, 1, True) == ["qzR_cov:R_1"]
    assert flags(sphere, 2, 2, True) == ["qzR_cov:R_1"]
    assert flags(sphere, 2.01, 2, True) == ["qzR_cov:R_1", "qzR_fail:R_1"]
    assert flags(sphere, 2.01, 3, True) == []
    assert flags(sphere, 1.3, 2, False) == []
    assert refine.interface_flags(sphere, 1.71) == ["qzR:R_1"]
    for qz in (None, 0, np.nan):
        assert flags(sphere, qz, 1, True) == []


@pytest.mark.parametrize("family", ["vertical_cylinder", "cylinder"])
def test_shape_warning_and_numbering(family):
    model = fm.FitModel((fm.new_component(family, radius=2, heights=True), fm.new_component("sphere", radius=1)))
    for aspect in (0.8, 1.25):
        m = model.with_parameter((0, "h"), value=4 * aspect)
        flags = reporting.grazing_flags(m, 0.7, 1, True)
        assert "cyl_sphere:R_2" in flags
        assert ("qzR_cov:R_2" in flags) == (family == "cylinder")
        assert reporting.grazing_flags(m, 0.6, 1, True) == []
        assert fit_names.clean_flags(flags, {"Int_2": 0}) == ("absent:2",)
    for height in (3.19, 5.01):
        m = model.with_parameter((0, "h"), value=height)
        assert "cyl_sphere:R_2" not in reporting.grazing_flags(m, 0.7, 1, True)
    upright = fm.FitModel((fm.new_component("vertical_cylinder", radius=2),))
    assert reporting.grazing_flags(upright, 1, 2) == []
    assert reporting.grazing_flags(upright, 1, 2, known_heights={"1": 4}) == ["cyl_sphere:R_1"]


def test_reflection_phase_summary_respects_the_branch_of_the_fit():
    from src.gimap.features.fitting.domain.posterior import encoding as E

    angle = np.deg2rad([178, 179, -179, -178])
    samples = np.zeros((4, E.DIMS))
    samples[:, E.GLOBAL.start + 4] = .8 * np.cos(angle)
    samples[:, E.GLOBAL.start + 5] = .8 * np.sin(angle)
    local = SimpleNamespace(samples=samples, tempered=lambda ess: np.full(4, .25))
    quantiles = reporting.reflection_quantiles(local, np.pi, 50)
    assert quantiles["refl_amp"]["median"] == pytest.approx(.8)
    assert quantiles["refl_phase"]["median"] == pytest.approx(np.pi)
    assert quantiles["refl_phase"]["high68"] - quantiles["refl_phase"]["low68"] < .08


def test_density_uses_fitted_range_and_strict_ratio():
    for n, warn in [(397, False), (150, True), (595, False), (596, True)]:
        d = reporting.density_record(np.linspace(0.02, 2, n))
        assert d["point_count"] == n and d["points_per_decade"] == n / 2
        assert d["validation_points_per_decade"] == 198.5 and d["density_notice"] == warn
        assert bool(fit_names.diagnostic_lines(d)) == warn


def test_bootstrap_normalizes_all_modes_with_independent_replicates():
    weights = np.linspace(0.1, 1.9, 24)
    groups = [0, 0, 2]
    entries = [(SimpleNamespace(index=group), None, None, None,
                SimpleNamespace(log_weights=np.log(weights) + 900, log_evidence=900.)) for group in groups]
    records = reporting.comparison_diagnostics(entries, False, {0: 2 / 3, 2: 1 / 3})
    rng = np.random.default_rng(reporting.BOOTSTRAP_SEED)
    draws = np.array([rng.choice(weights, (200, 24)).mean(axis=1) for _ in groups])
    expected = np.quantile(draws[:2].sum(axis=0) / draws.sum(axis=0), [0.16, 0.84])
    for record in records[:2]:
        assert record["composition_probability_mc68"] == pytest.approx(expected)
    assert records[0]["log_evidence_mc_error"] == pytest.approx(np.std(np.log(draws[0]), ddof=1))
    assert records == reporting.comparison_diagnostics(entries, False, {0: 2 / 3, 2: 1 / 3})
    entries[0] = (SimpleNamespace(index=0, log_weights=np.zeros(24), log_evidence=0.), *entries[0][1:4], None)
    fallbacks = reporting.comparison_diagnostics(entries, True, {0: .8, 2: .2})
    assert fallbacks[0]["log_evidence_mc_error"] == 0 and fallbacks[0]["log_evidence_source"] == "network"
    assert all("composition_probability_mc68" not in r and r["compared_by"] == "bic" for r in fallbacks)


def test_bic_record_csv_project_and_report_preserve_separate_estimates(tmp_path):
    from src.gimap.features.fitting.application import fit_report
    from tests.test_fit_page import _curve_file, _page

    page = _page()
    try:
        assert page.open_curve(_curve_file(tmp_path))
        model = fm.FitModel((fm.new_component("sphere", radius=4),))
        raw = {"median": 5., "low68": 4.5, "high68": 6., "low95": 4.1, "high95": 7.}
        solution = Solution(model, 1.2, "Predict parameters · sphere", probability=.7, ess=25,
                            flags=("bic", "qzR_cov:R_1"), intervals=(("R_1", (4, 4, 3, 5, 2, 6, 1)),),
                            posterior_summary={"R_1": raw}, diagnostics={"compared_by": "bic", "interval_kind": "bic",
                                "posterior_kind": "tempered", **reporting.density_record(np.linspace(.1, 1, 20))})
        page.session.solutions = [solution]
        page.session.solutions_selection = page.session.selection()
        page.set_model(model)
        page.render_results()
        assert page.parameters_table.horizontalHeaderItem(3).text() == "posterior median"
        row = page._row_paths.index((0, "R"))
        assert page.parameters_table.item(row, 3).text() == "5"
        assert "5.00 [4.50, 6.00]" in page.parameters_table.item(row, 3).toolTip()
        for flag in ("qzR_cov:R_1", "qzR_fail:R_1", "cyl_sphere:R_1"):
            page.session.solutions = [replace(solution, flags=("bic", flag))]
            page.render_results()
            assert "Needs attention" in page.quality_label.text()
        page.session.solutions = [solution]
        page.render_results()
        assert page.solutions_table.horizontalHeaderItem(5).text() == "BIC weight"
        assert "BIC weight" in predict_texts.probability_line(solution, [solution])
        record = solution_export.solution_record(solution, 1, "sphere", True)
        assert record["bic_weight"] == .7 and "probability" not in record and record["ess"] == 25
        param = next(p for p in record["parameters"] if p["name"] == "R")
        assert (param["least_squares_value"], param["posterior_median"], param["posterior_low68"]) == (4, 5, 4.5)
        assert param["posterior_high95"] == 7 and param["reported_interval_kind"] == "bic"
        assert fit_names.reported_interval_kind(("bic",), None, "tempered") == "tempered"
        assert fit_names.reported_interval_kind(("bic",), None) == "network_proposal"
        old = replace(solution, posterior_summary={}, diagnostics={}, flags=(), intervals=(("R_1", (4, 5, 4, 6, 3, 7, 1)),))
        assert solution_export.posterior_by_path(old)[(0, "R")] == {"median": 5}
        assert not solution_export.posterior_by_path(replace(old, flags=("bic",)))
        columns, rows = solution_export.solutions_table([solution], model)
        assert "probability" not in columns and "bic_weight" in columns and len(columns) == len(rows[0])
        assert project.solution_from_dict(project.solution_to_dict(solution)) == solution
        for fmt in fit_report.FORMATS:
            report = page._report_text(fmt)
            assert "BIC weight 0.7" in report and "Posterior median" in report and "198.5" in report
    finally:
        page.dispose()
